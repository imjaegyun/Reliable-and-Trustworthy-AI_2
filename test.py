#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def resolve_path(base_dir: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def require_path(path: Path, label: str, *, is_dir: bool = False) -> None:
    if is_dir and not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")
    if not is_dir and not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepXplore differential testing for CIFAR-10 ResNet50 models.",
    )
    parser.add_argument(
        "--deepxplore-dir",
        default="../deepxplore",
        help="Path to the local DeepXplore clone.",
    )
    parser.add_argument(
        "--model-a",
        default=None,
        help="Path to the first CIFAR-10 ResNet50 checkpoint.",
    )
    parser.add_argument(
        "--model-b",
        default=None,
        help="Path to the second CIFAR-10 ResNet50 checkpoint.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for generated images and summaries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and write run_config.json without running DeepXplore.",
    )
    return parser.parse_args()


def validate_deepxplore_dir(deepxplore_dir: Path) -> None:
    require_path(deepxplore_dir, "DeepXplore", is_dir=True)

    expected_files = [
        deepxplore_dir / "README.md",
        deepxplore_dir / "MNIST" / "gen_diff.py",
        deepxplore_dir / "ImageNet" / "gen_diff.py",
    ]
    missing = [str(path) for path in expected_files if not path.exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(f"DeepXplore clone looks incomplete:\n  - {formatted}")


def write_run_config(
    results_dir: Path,
    deepxplore_dir: Path,
    model_paths: list[Path],
    dry_run: bool,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deepxplore_dir": str(deepxplore_dir),
        "models": [str(path) for path in model_paths],
        "classes": CIFAR10_CLASSES,
        "dry_run": dry_run,
    }
    output_path = results_dir / "run_config.json"
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return output_path


def run_deepxplore(
    deepxplore_dir: Path,
    model_paths: list[Path],
    results_dir: Path,
) -> int:
    raise NotImplementedError(
        "DeepXplore/CIFAR-10 execution is not implemented yet. "
        "Use --dry-run for the initial check, then implement run_deepxplore()."
    )


def main() -> int:
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent

    deepxplore_dir = resolve_path(repo_dir, args.deepxplore_dir)
    results_dir = resolve_path(repo_dir, args.results_dir)
    model_paths = [
        resolve_path(repo_dir, path)
        for path in [args.model_a, args.model_b]
        if path is not None
    ]

    validate_deepxplore_dir(deepxplore_dir)
    config_path = write_run_config(
        results_dir=results_dir,
        deepxplore_dir=deepxplore_dir,
        model_paths=model_paths,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"Dry run complete. Wrote {config_path}")
        return 0

    if len(model_paths) < 2:
        print("Error: provide --model-a and --model-b for the real run.", file=sys.stderr)
        return 2

    for index, model_path in enumerate(model_paths, start=1):
        require_path(model_path, f"model {index}")

    return run_deepxplore(
        deepxplore_dir=deepxplore_dir,
        model_paths=model_paths,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
