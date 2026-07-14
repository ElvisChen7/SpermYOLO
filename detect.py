"""Run SpermYOLO detection on one image, a directory, video, or stream."""

from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", required=True, help="image, image directory, video, or stream URL")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    args = parser.parse_args()

    YOLO(str(args.weights.resolve())).predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )


if __name__ == "__main__":
    main()
