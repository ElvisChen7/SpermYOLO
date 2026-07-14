# SpermYOLO

Official implementation of SpermYOLO, a YOLOv11-based detector for sperm microscopy images. The model introduces C3k2-IDB, D2SEM, MFM, and the DESD Head.

This repository supports object detection training, evaluation, and prediction only. It is derived from [Ultralytics](https://github.com/ultralytics/ultralytics) and released under AGPL-3.0.

## Environment

The experiments were conducted with Python 3.11, PyTorch 2.2.1, torchvision 0.17.1, and CUDA 12.1. Install a PyTorch build appropriate for your system, then install the remaining packages:

```bash
python -m pip install numpy opencv-python pillow pyyaml requests scipy pandas matplotlib psutil tqdm seaborn dill
```

## Data preparation

Dataset files are not included. Place images and YOLO-format labels as follows:

```text
datasets/
└── SVIA/                         # or SDTB/
    ├── SVIA.yaml                 # or SDTB.yaml
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

Each label file must share its image filename and use the format `class x_center y_center width height`, with normalized coordinates. The original split manifests are retained as `train.txt`, `val.txt`, and `test.txt` under each dataset directory.

## Training

```bash
python train_SVIA.py --epochs 200 --batch 16 --device 0
python train_SDTB.py --epochs 400 --batch 16 --device 0
```

To train the YOLOv11 baseline, pass its YAML explicitly:

```bash
python train_SVIA.py --model ultralytics/cfg/models/11/yolo11.yaml --device 0
```

## Evaluation and prediction

```bash
python val_SVIA.py --weights runs/SVIA-SpermYOLO/weights/best.pt --split test --device 0
python val_SDTB.py --weights runs/SDTB-SpermYOLO/weights/best.pt --split test --device 0

python detect.py --weights runs/SVIA-SpermYOLO/weights/best.pt --source datasets/SVIA/images/test --save-txt --save-conf --device 0
```

## Model configuration

```python
from ultralytics import YOLO

baseline = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
spermyolo = YOLO("ultralytics/cfg/models/11/yolo11-MFM-C3k2-IDB-D2SEM-DESD-Head.yaml")
```

## License

SpermYOLO is released under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE) for the full text.
