"""PyTorch-only inference wrapper for the SpermYOLO detection runtime."""

from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.utils.checks import check_yaml


def check_class_names(names):
    """Normalize detection class names to an integer-keyed dictionary."""
    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
        if names and max(names) >= len(names):
            raise KeyError(f"Invalid class indices: {min(names)}-{max(names)} for {len(names)} classes.")
    return names


def default_class_names(data=None):
    """Return names from a data YAML, or generic detection class names."""
    if data:
        try:
            from ultralytics.utils import yaml_load

            return check_class_names(yaml_load(check_yaml(data))["names"])
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}


class AutoBackend(nn.Module):
    """Run an in-memory detection model or a local PyTorch ``.pt`` checkpoint."""

    @torch.no_grad()
    def __init__(self, weights, device=torch.device("cpu"), dnn=False, data=None, fp16=False, batch=1, fuse=True, verbose=True):
        super().__init__()
        if dnn:
            raise ValueError("OpenCV-DNN inference is not included in this PyTorch-only runtime.")
        if isinstance(weights, nn.Module):
            model = weights.to(device)
        else:
            weight = Path(weights[0] if isinstance(weights, list) else weights)
            if weight.suffix != ".pt":
                raise ValueError("Only local PyTorch .pt checkpoints are supported for inference.")
            from ultralytics.nn.tasks import attempt_load_weights

            model = attempt_load_weights(weight, device=device, inplace=True, fuse=fuse)
        if fuse and hasattr(model, "fuse"):
            model = model.fuse(verbose=verbose)

        self.model = model.eval()
        self.device = device
        self.fp16 = bool(fp16 and device.type != "cpu")
        self.model.half() if self.fp16 else self.model.float()
        self.stride = max(int(self.model.stride.max()), 32)
        self.names = check_class_names(getattr(self.model, "names", None) or default_class_names(data))
        self.pt, self.jit, self.engine, self.triton, self.nn_module = True, False, False, False, True
        self.batch_size, self.metadata = batch, {}

    def forward(self, im, augment=False, visualize=False, embed=None, **kwargs):
        """Execute the wrapped PyTorch detection model."""
        return self.model(im, augment=augment, visualize=visualize, embed=embed)

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """Run a single warmup pass on non-CPU devices."""
        if self.device.type != "cpu":
            self.forward(torch.empty(*imgsz, device=self.device, dtype=torch.half if self.fp16 else torch.float))

    def from_numpy(self, x):
        """Move a NumPy array to the selected device."""
        return torch.from_numpy(x).to(self.device)
