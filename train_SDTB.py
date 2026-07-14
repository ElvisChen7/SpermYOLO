"""Train SpermYOLO on an SDTB-format detection dataset."""

from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
SPERM_YAML = ROOT / "ultralytics/cfg/models/11/yolo11-MFM-C3k2-IDB-D2SEM-DESD-Head.yaml"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=SPERM_YAML, help="model YAML or .pt checkpoint")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/SDTB/SDTB.yaml")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="for example 0, 0,1, or cpu")
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--name", default="SDTB-SpermYOLO")
    args = parser.parse_args()

    YOLO(str(args.model.resolve())).train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer="SGD",
        patience=args.epochs,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
    )


if __name__ == "__main__":
    main()
