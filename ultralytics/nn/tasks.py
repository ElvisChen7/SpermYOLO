"""Detection-only model definitions and YAML parser for SpermYOLO."""

import ast
import re
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.extra_modules import C3k2_IDB, D2SEM, DESD_Head, MFM
from ultralytics.nn.modules import C2PSA, C3k2, Concat, Conv, DFL, Detect, DWConv, SPPF
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, yaml_load
from ultralytics.utils.checks import check_suffix, check_yaml
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
)

DETECT_CLASS = (Detect, DESD_Head)


class BaseModel(nn.Module):
    """Base model shared by the released detection architectures."""

    def forward(self, x, *args, **kwargs):
        return self.loss(x, *args, **kwargs) if isinstance(x, dict) else self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        return self._predict_augment(x) if augment else self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        y, embeddings = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1))
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Run YOLO-style augmented inference for the standard detection model."""
        if self.__class__.__name__ != "DetectionModel" or isinstance(self.model[-1], DESD_Head):
            LOGGER.warning("WARNING: augmented prediction is unavailable for this model; using single-scale inference.")
            return self._predict_once(x)
        image_size, scales, flips, predictions = x.shape[-2:], (1, 0.83, 0.67), (None, 3, None), []
        for scale, flip in zip(scales, flips):
            xi = scale_img(x.flip(flip) if flip else x, scale, gs=int(self.stride.max()))
            yi = self._predict_once(xi)[0]
            yi[:, :4] /= scale
            px, py, pwh, pcls = yi.split((1, 1, 2, yi.shape[1] - 4), 1)
            if flip == 2:
                py = image_size[0] - py
            elif flip == 3:
                px = image_size[1] - px
            predictions.append(torch.cat((px, py, pwh, pcls), 1))
        return torch.cat(predictions, -1), None

    def fuse(self, verbose=True):
        """Fuse supported convolution and normalization layers for inference."""
        if self.is_fused():
            return self
        for module in self.model.modules():
            if isinstance(module, (Conv, DWConv)) and hasattr(module, "bn"):
                module.conv = fuse_conv_and_bn(module.conv, module.bn)
                delattr(module, "bn")
                module.forward = module.forward_fuse
            elif hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
        self.info(verbose=verbose)
        return self

    def is_fused(self, thresh=10):
        norm_layers = tuple(value for name, value in nn.__dict__.items() if "Norm" in name)
        return sum(isinstance(module, norm_layers) for module in self.modules()) < thresh

    def info(self, detailed=False, verbose=True, imgsz=640):
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        self = super()._apply(fn)
        head = self.model[-1]
        if isinstance(head, DETECT_CLASS):
            head.stride = fn(head.stride)
            head.anchors = fn(head.anchors)
            head.strides = fn(head.strides)
        return self

    def load(self, weights, verbose=True):
        model = weights["model"] if isinstance(weights, dict) else weights
        state_dict = intersect_dicts(model.float().state_dict(), self.state_dict())
        self.load_state_dict(state_dict, strict=False)
        if verbose:
            LOGGER.info(f"Transferred {len(state_dict)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        return self.criterion(self.forward(batch["img"]) if preds is None else preds, batch)

    def init_criterion(self):
        raise NotImplementedError


class DetectionModel(BaseModel):
    """YOLOv11/SpermYOLO object-detection model."""

    def __init__(self, cfg="yolo11.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: str(i) for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)

        head = self.model[-1]
        if not isinstance(head, DETECT_CLASS):
            raise TypeError("The minimal runtime supports detection YAML files only.")
        stride_input = 640
        head.inplace = self.inplace
        head.stride = torch.tensor(
            [stride_input / output.shape[-2] for output in self.forward(torch.zeros(2, ch, stride_input, stride_input))]
        )
        self.stride = head.stride
        head.bias_init()
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def init_criterion(self):
        return v8DetectionLoss(self)


class Ensemble(nn.ModuleList):
    """Detection-model ensemble."""

    def forward(self, x, augment=False, profile=False, visualize=False):
        outputs = [module(x, augment, profile, visualize)[0] for module in self]
        return torch.cat(outputs, 2), None


def torch_safe_load(weight, safe_only=False):
    """Load a local Ultralytics-format PyTorch checkpoint without auto-installing packages."""
    check_suffix(file=weight, suffix=".pt")
    file = Path(weight)
    if not file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {file}")
    try:
        checkpoint = torch.load(file, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions predating weights_only
        checkpoint = torch.load(file, map_location="cpu")
    if not isinstance(checkpoint, dict):
        checkpoint = {"model": checkpoint.model}
    return checkpoint, str(file)


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Load one or more local detection checkpoints."""
    ensemble = Ensemble()
    for weight in weights if isinstance(weights, list) else [weights]:
        checkpoint, path = torch_safe_load(weight)
        args = {**DEFAULT_CFG_DICT, **checkpoint.get("train_args", {})}
        model = (checkpoint.get("ema") or checkpoint["model"]).to(device).float()
        model.args, model.pt_path, model.task = args, path, "detect"
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])
        ensemble.append(model.fuse().eval() if fuse else model.eval())
    for module in ensemble.modules():
        if hasattr(module, "inplace"):
            module.inplace = inplace
        elif isinstance(module, nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
            module.recompute_scale_factor = None
    if len(ensemble) == 1:
        return ensemble[0]
    for key in ("names", "nc", "yaml"):
        setattr(ensemble, key, getattr(ensemble[0], key))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([model.stride.max() for model in ensemble])))].stride
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Load one local detection checkpoint and its metadata."""
    checkpoint, path = torch_safe_load(weight)
    model = (checkpoint.get("ema") or checkpoint["model"]).to(device).float()
    args = {**DEFAULT_CFG_DICT, **checkpoint.get("train_args", {})}
    model.args = {key: value for key, value in args.items() if key in DEFAULT_CFG_KEYS}
    model.pt_path, model.task = path, "detect"
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])
    model = model.fuse().eval() if fuse else model.eval()
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = inplace
        elif isinstance(module, nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
            module.recompute_scale_factor = None
    return model, checkpoint


def parse_model(d, ch, verbose=True):
    """Parse either of the two supported detection YAML formats."""
    max_channels = float("inf")
    nc, act, scales = (d.get(key) for key in ("nc", "activation", "scales"))
    depth, width = (d.get(key, 1.0) for key in ("depth_multiple", "width_multiple"))
    scale = d.get("scale", "")
    if scales:
        scale = scale or next(iter(scales))
        depth, width, max_channels = scales[scale]
    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    module_registry = {
        "Conv": Conv,
        "C3k2": C3k2,
        "C3k2_IDB": C3k2_IDB,
        "SPPF": SPPF,
        "C2PSA": C2PSA,
        "D2SEM": D2SEM,
        "Concat": Concat,
        "Detect": Detect,
        "DESD_Head": DESD_Head,
        "MFM": MFM,
    }
    channel_modules = {Conv, C3k2, C3k2_IDB, SPPF, C2PSA, D2SEM}
    repeat_modules = {C3k2, C3k2_IDB, C2PSA, D2SEM}
    channels = [ch]
    layers, save = [], []
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<32}{'arguments':<30}")
    for i, (source, repeats, module, args) in enumerate(d["backbone"] + d["head"]):
        if isinstance(module, str):
            module = getattr(nn, module[3:]) if module.startswith("nn.") else module_registry[module]
        args = list(args)
        for index, value in enumerate(args):
            if isinstance(value, str):
                args[index] = nc if value == "nc" else ast.literal_eval(value) if value in {"True", "False", "None"} else value
        repeats_scaled = max(round(repeats * depth), 1) if repeats > 1 else repeats
        if module in channel_modules:
            c1, c2 = channels[source], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if module in repeat_modules:
                args.insert(2, repeats_scaled)
                repeats_scaled = 1
            if module in {C3k2, C3k2_IDB} and scale in "mlx":
                args[3] = True
        elif module is nn.Upsample:
            c2 = channels[source]
        elif module is Concat:
            c2 = sum(channels[index] for index in source)
        elif module in DETECT_CLASS:
            args.append([channels[index] for index in source])
            if module is DESD_Head:
                args[1] = make_divisible(min(args[1], max_channels) * width, 8)
            c2 = channels[source[-1]]
        elif module is MFM:
            c1 = [channels[index] for index in source]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        else:
            raise ValueError(f"Unsupported module in minimal runtime: {module}")

        layer = nn.Sequential(*(module(*args) for _ in range(repeats_scaled))) if repeats_scaled > 1 else module(*args)
        layer.i, layer.f, layer.type = i, source, module.__name__
        layer.np = sum(parameter.numel() for parameter in layer.parameters())
        if verbose:
            LOGGER.info(f"{i:>3}{str(source):>20}{repeats:>3}{layer.np:10.0f}  {layer.type:<32}{str(args):<30}")
        save.extend(index % i for index in ([source] if isinstance(source, int) else source) if index != -1)
        layers.append(layer)
        if i == 0:
            channels = []
        channels.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a supported YOLOv11/SpermYOLO YAML file."""
    path = Path(path)
    yaml_file = check_yaml(path, hard=False) or check_yaml(path)
    data = yaml_load(yaml_file)
    data["scale"] = guess_model_scale(path)
    data["yaml_file"] = str(path)
    return data


def guess_model_scale(model_path):
    match = re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem)
    return match.group(1) if match else ""


def guess_model_task(model):
    """The public runtime supports detection models only."""
    return "detect"
