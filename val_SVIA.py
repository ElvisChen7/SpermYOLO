"""Validate a SpermYOLO detection checkpoint on SVIA."""

from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/SVIA/SVIA.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--name", default="SVIA-val")
    args = parser.parse_args()

    metrics = YOLO(str(args.weights.resolve())).val(
        data=str(args.data.resolve()),
        split=args.split,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
    )
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
