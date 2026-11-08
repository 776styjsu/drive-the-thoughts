"""CLI with lightweight discovery; torch is imported only for model operations."""

import argparse
import json
from pathlib import Path

from .data import discover, prepare, read_json, validate_manifest, write_json


def parser():
    root = argparse.ArgumentParser(description="SelfOracle adaptation for AlpaSim")
    sub = root.add_subparsers(dest="command", required=True)
    discovery = sub.add_parser("discover", help="Inventory extracted frames; curate splits before prepare")
    discovery.add_argument("--root", type=Path, action="append", required=True)
    discovery.add_argument("--camera", default="camera_front_wide_120fov.jpg")
    discovery.add_argument("--holdout-benchmark", type=Path)
    discovery.add_argument("--output", type=Path, required=True)
    preparation = sub.add_parser("prepare", help="Validate curated inventory and hash selected images")
    preparation.add_argument("--inventory", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    validation = sub.add_parser("validate", help="Check file availability and scene-disjoint splits")
    validation.add_argument("--manifest", type=Path, required=True)
    for command in ("train", "calibrate", "predict"):
        stage = sub.add_parser(command)
        stage.add_argument("--manifest", type=Path, required=True)
        stage.add_argument("--output", type=Path, required=True)
        stage.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        stage.add_argument("--batch-size", type=int, default=32)
        if command == "train":
            stage.add_argument("--model", choices=["sae", "vae"], default="sae")
            stage.add_argument("--epochs", type=int, default=20)
            stage.add_argument("--learning-rate", type=float, default=1e-3)
            stage.add_argument("--seed", type=int, default=0)
            stage.add_argument("--height", type=int, default=80)
            stage.add_argument("--width", type=int, default=160)
        else:
            stage.add_argument("--checkpoint", type=Path, required=True)
        if command == "calibrate":
            stage.add_argument("--epsilon", type=float, default=0.05)
            stage.add_argument("--window", type=int, default=5)
            stage.add_argument("--max-gap-us", type=int, default=150_000)
        if command == "predict":
            stage.add_argument("--calibration", type=Path, required=True)
    return root


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == "discover":
            if args.output.exists():
                raise ValueError(f"Refusing to overwrite curated inventory: {args.output}")
            write_json(args.output, discover(args.root, args.camera, args.holdout_benchmark))
        elif args.command == "prepare":
            if args.output.exists():
                raise ValueError(f"Output already exists: {args.output}")
            manifest = prepare(args.inventory)
            write_json(args.output, manifest)
            print(json.dumps(manifest["summary"], indent=2))
        elif args.command == "validate":
            print(json.dumps(validate_manifest(read_json(args.manifest)), indent=2))
        else:
            from . import pipeline
            getattr(pipeline, args.command)(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    main()

