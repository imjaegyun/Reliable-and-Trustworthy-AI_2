from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from models import CIFAR10_CLASSES, load_checkpoint


@dataclass
class CoverageState:
    covered: dict[str, torch.Tensor]
    total: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepxplore-dir", default="../deepxplore")
    parser.add_argument("--data-dir", default="../../assignment1/data")
    parser.add_argument("--model-a", default="models/resnet50_cifar10_seed1.pt")
    parser.add_argument("--model-b", default="models/resnet50_cifar10_seed2.pt")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.12)
    parser.add_argument("--weight-diff", type=float, default=1.0)
    parser.add_argument("--weight-nc", type=float, default=0.2)
    parser.add_argument("--coverage-threshold", type=float, default=0.75)
    parser.add_argument("--coverage-sweep-thresholds", default="0.2,0.5,0.75,0.9")
    parser.add_argument("--disagreement-sweep-thresholds", default="0.2,0.5,0.75,0.9")
    parser.add_argument("--max-visualizations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_path(base_dir: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def require_path(path: Path, label: str, is_dir: bool = False) -> None:
    if is_dir and not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")
    if not is_dir and not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for item in value.split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    return sorted(set(thresholds))


def threshold_key(threshold: float) -> str:
    return f"{threshold:g}"


def threshold_filename(threshold: float) -> str:
    return threshold_key(threshold).replace(".", "p")


def resolve_device(device_name: str) -> torch.device:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuda_available = torch.cuda.is_available()
    if device_name == "auto":
        return torch.device("cuda" if cuda_available else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return device


def validate_deepxplore_dir(deepxplore_dir: Path) -> None:
    require_path(deepxplore_dir, "DeepXplore", is_dir=True)
    for relative_path in ["README.md", "ImageNet/gen_diff.py", "ImageNet/utils.py"]:
        require_path(deepxplore_dir / relative_path, f"DeepXplore {relative_path}")


def make_loader(args: argparse.Namespace, repo_dir: Path) -> DataLoader:
    data_dir = resolve_path(repo_dir, args.data_dir)
    dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=args.download,
        transform=transforms.ToTensor(),
    )
    limit = min(args.seeds, len(dataset))
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def coverage_layers(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    return {
        "layer1": model.backbone.layer1,
        "layer2": model.backbone.layer2,
        "layer3": model.backbone.layer3,
        "layer4": model.backbone.layer4,
    }


def init_coverage(model: torch.nn.Module, device: torch.device) -> CoverageState:
    channels = {
        "layer1": model.backbone.layer1[-1].conv3.out_channels,
        "layer2": model.backbone.layer2[-1].conv3.out_channels,
        "layer3": model.backbone.layer3[-1].conv3.out_channels,
        "layer4": model.backbone.layer4[-1].conv3.out_channels,
    }
    covered = {
        name: torch.zeros(count, dtype=torch.bool, device=device)
        for name, count in channels.items()
    }
    return CoverageState(covered=covered, total=sum(channels.values()))


def collect_activations(model: torch.nn.Module, images: torch.Tensor) -> dict[str, torch.Tensor]:
    activations: dict[str, torch.Tensor] = {}
    handles = []
    for name, layer in coverage_layers(model).items():
        handles.append(layer.register_forward_hook(lambda _, __, output, key=name: activations.__setitem__(key, output)))
    model(images)
    for handle in handles:
        handle.remove()
    return activations


def update_coverage(
    state: CoverageState,
    activations: dict[str, torch.Tensor],
    threshold: float,
) -> None:
    for name, output in activations.items():
        values = output.detach().flatten(2).mean(dim=(0, 2))
        minimum = values.min()
        maximum = values.max()
        scaled = (values - minimum) / (maximum - minimum + 1e-8)
        state.covered[name] |= scaled > threshold


def coverage_fraction(state: CoverageState) -> float:
    covered = sum(mask.sum().item() for mask in state.covered.values())
    return covered / max(state.total, 1)


def update_sweep(
    sweep_states: list[tuple[float, CoverageState, CoverageState]],
    activations_a: dict[str, torch.Tensor],
    activations_b: dict[str, torch.Tensor],
) -> None:
    for threshold, state_a, state_b in sweep_states:
        update_coverage(state_a, activations_a, threshold)
        update_coverage(state_b, activations_b, threshold)


def pick_uncovered(state: CoverageState) -> tuple[str, int]:
    candidates = []
    for name, mask in state.covered.items():
        indices = torch.where(~mask)[0].tolist()
        candidates.extend((name, index) for index in indices)
    if not candidates:
        for name, mask in state.covered.items():
            candidates.extend((name, index) for index in range(mask.numel()))
    return random.choice(candidates)


def selected_activation(
    activations: dict[str, torch.Tensor],
    layer_name: str,
    channel_index: int,
) -> torch.Tensor:
    output = activations[layer_name]
    if output.ndim == 4:
        return output[:, channel_index].mean()
    return output[:, channel_index].mean()


def predict(model: torch.nn.Module, images: torch.Tensor) -> tuple[int, float]:
    with torch.no_grad():
        probabilities = F.softmax(model(images), dim=1)
        confidence, label = probabilities.max(dim=1)
    return label.item(), confidence.item()


def save_visualization(
    path: Path,
    original: torch.Tensor,
    generated: torch.Tensor,
    true_label: int,
    predictions: dict[str, tuple[int, float]],
) -> None:
    original_np = original.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    generated_np = generated.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    figure, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(np.clip(original_np, 0, 1))
    axes[0].set_title(f"seed: {CIFAR10_CLASSES[true_label]}")
    axes[0].axis("off")
    axes[1].imshow(np.clip(generated_np, 0, 1))
    title_lines = []
    for model_name, (label, confidence) in predictions.items():
        title_lines.append(f"{model_name}: {CIFAR10_CLASSES[label]} ({confidence:.2f})")
    axes[1].set_title("\n".join(title_lines))
    axes[1].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_deepxplore(
    args: argparse.Namespace,
    repo_dir: Path,
    coverage_threshold: float | None = None,
    write_summary: bool = True,
    save_visualizations: bool = True,
    visualization_prefix: str = "disagreement",
    progress_desc: str = "deepxplore",
) -> dict:
    if args.batch_size != 1:
        raise ValueError("batch-size must be 1 for gradient-based input generation")

    if coverage_threshold is None:
        coverage_threshold = args.coverage_threshold
    set_seed(args.seed)
    device = resolve_device(args.device)
    deepxplore_dir = resolve_path(repo_dir, args.deepxplore_dir)
    model_a_path = resolve_path(repo_dir, args.model_a)
    model_b_path = resolve_path(repo_dir, args.model_b)
    results_dir = resolve_path(repo_dir, args.results_dir)
    validate_deepxplore_dir(deepxplore_dir)
    if write_summary or save_visualizations:
        results_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        payload = {
            "deepxplore_dir": str(deepxplore_dir),
            "model_a": str(model_a_path),
            "model_b": str(model_b_path),
            "results_dir": str(results_dir),
            "dry_run": True,
        }
        if write_summary:
            write_json(results_dir / "run_config.json", payload)
        return payload

    require_path(model_a_path, "model A")
    require_path(model_b_path, "model B")
    model_a = load_checkpoint(str(model_a_path), device)
    model_b = load_checkpoint(str(model_b_path), device)
    coverage_a = init_coverage(model_a, device)
    coverage_b = init_coverage(model_b, device)
    sweep_thresholds = parse_thresholds(args.coverage_sweep_thresholds)
    if not any(abs(threshold - coverage_threshold) < 1e-9 for threshold in sweep_thresholds):
        sweep_thresholds.append(coverage_threshold)
        sweep_thresholds = sorted(set(sweep_thresholds))
    coverage_sweep = [
        (threshold, init_coverage(model_a, device), init_coverage(model_b, device))
        for threshold in sweep_thresholds
    ]
    loader = make_loader(args, repo_dir)
    disagreements = []
    saved = 0

    for sample_index, (images, labels) in enumerate(tqdm(loader, desc=progress_desc)):
        images = images.to(device)
        labels = labels.to(device)
        original = images.detach().clone()
        true_label = labels.item()
        label_a, conf_a = predict(model_a, images)
        label_b, conf_b = predict(model_b, images)
        activations_a = collect_activations(model_a, images)
        activations_b = collect_activations(model_b, images)
        update_coverage(coverage_a, activations_a, coverage_threshold)
        update_coverage(coverage_b, activations_b, coverage_threshold)
        update_sweep(coverage_sweep, activations_a, activations_b)

        if label_a != label_b:
            generated = images.detach()
        else:
            generated = images.detach().clone()
            target_label = label_a
            target_model = sample_index % 2
            for _ in range(args.iterations):
                generated.requires_grad_(True)
                generated.grad = None
                activations_a = collect_activations(model_a, generated)
                activations_b = collect_activations(model_b, generated)
                logits_a = model_a(generated)
                logits_b = model_b(generated)
                layer_a, channel_a = pick_uncovered(coverage_a)
                layer_b, channel_b = pick_uncovered(coverage_b)
                neuron_loss = selected_activation(activations_a, layer_a, channel_a)
                neuron_loss = neuron_loss + selected_activation(activations_b, layer_b, channel_b)
                if target_model == 0:
                    diff_loss = logits_b[:, target_label].mean() - logits_a[:, target_label].mean()
                else:
                    diff_loss = logits_a[:, target_label].mean() - logits_b[:, target_label].mean()
                loss = args.weight_diff * diff_loss + args.weight_nc * neuron_loss
                loss.backward()
                with torch.no_grad():
                    step = args.step * generated.grad.sign()
                    generated = generated + step
                    generated = torch.max(torch.min(generated, original + args.epsilon), original - args.epsilon)
                    generated = generated.clamp(0.0, 1.0).detach()
                label_a, conf_a = predict(model_a, generated)
                label_b, conf_b = predict(model_b, generated)
                activations_a = collect_activations(model_a, generated)
                activations_b = collect_activations(model_b, generated)
                update_coverage(coverage_a, activations_a, coverage_threshold)
                update_coverage(coverage_b, activations_b, coverage_threshold)
                update_sweep(coverage_sweep, activations_a, activations_b)
                if label_a != label_b:
                    break

        if label_a != label_b:
            record = {
                "sample_index": sample_index,
                "true_label": CIFAR10_CLASSES[true_label],
                "model_a": {
                    "label": CIFAR10_CLASSES[label_a],
                    "confidence": conf_a,
                },
                "model_b": {
                    "label": CIFAR10_CLASSES[label_b],
                    "confidence": conf_b,
                },
                "l_inf_delta": (generated - original).abs().max().item(),
            }
            disagreements.append(record)
            if save_visualizations and saved < args.max_visualizations:
                output_path = results_dir / f"{visualization_prefix}_{saved + 1:02d}.png"
                save_visualization(
                    output_path,
                    original,
                    generated,
                    true_label,
                    {
                        "model A": (label_a, conf_a),
                        "model B": (label_b, conf_b),
                    },
                )
                record["visualization"] = str(output_path)
                saved += 1

    coverage_by_threshold = {}
    for threshold, state_a, state_b in coverage_sweep:
        coverage_a_value = coverage_fraction(state_a)
        coverage_b_value = coverage_fraction(state_b)
        coverage_by_threshold[threshold_key(threshold)] = {
            "model_a": coverage_a_value,
            "model_b": coverage_b_value,
            "average": (coverage_a_value + coverage_b_value) / 2,
        }

    summary = {
        "num_seeds": args.seeds,
        "num_disagreements": len(disagreements),
        "coverage": {
            "model_a": coverage_fraction(coverage_a),
            "model_b": coverage_fraction(coverage_b),
            "average": (coverage_fraction(coverage_a) + coverage_fraction(coverage_b)) / 2,
        },
        "coverage_by_threshold": coverage_by_threshold,
        "disagreements": disagreements,
        "settings": {
            "iterations": args.iterations,
            "step": args.step,
            "epsilon": args.epsilon,
            "weight_diff": args.weight_diff,
            "weight_nc": args.weight_nc,
            "coverage_threshold": coverage_threshold,
            "coverage_sweep_thresholds": sweep_thresholds,
        },
    }
    if write_summary:
        write_json(results_dir / "summary.json", summary)
    return summary


def run_disagreement_threshold_sweep(
    args: argparse.Namespace,
    repo_dir: Path,
    base_summary: dict | None = None,
) -> dict:
    thresholds = parse_thresholds(args.disagreement_sweep_thresholds)
    if not any(abs(threshold - args.coverage_threshold) < 1e-9 for threshold in thresholds):
        thresholds.append(args.coverage_threshold)
        thresholds = sorted(set(thresholds))

    results = {}
    for threshold in thresholds:
        if base_summary is not None and abs(threshold - args.coverage_threshold) < 1e-9:
            summary = base_summary
        else:
            summary = run_deepxplore(
                args,
                repo_dir,
                coverage_threshold=threshold,
                write_summary=False,
                save_visualizations=True,
                visualization_prefix=f"disagreement_t{threshold_filename(threshold)}",
                progress_desc=f"threshold {threshold_key(threshold)}",
            )
        num_seeds = summary["num_seeds"]
        num_disagreements = summary["num_disagreements"]
        visualizations = [
            record["visualization"]
            for record in summary["disagreements"][: args.max_visualizations]
            if "visualization" in record
        ]
        results[threshold_key(threshold)] = {
            "num_seeds": num_seeds,
            "num_disagreements": num_disagreements,
            "disagreement_rate": num_disagreements / max(num_seeds, 1),
            "visualizations": visualizations,
        }

    return {
        "disagreement_by_threshold": results,
        "settings": {
            "thresholds": thresholds,
            "seeds": args.seeds,
            "iterations": args.iterations,
            "step": args.step,
            "epsilon": args.epsilon,
            "weight_diff": args.weight_diff,
            "weight_nc": args.weight_nc,
        },
    }


def main() -> int:
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    summary = run_deepxplore(args, repo_dir)
    if not summary.get("dry_run"):
        results_dir = resolve_path(repo_dir, args.results_dir)
        disagreement_sweep = run_disagreement_threshold_sweep(args, repo_dir, summary)
        write_json(results_dir / "disagreement_by_threshold.json", disagreement_sweep)
        summary["disagreement_by_threshold"] = disagreement_sweep["disagreement_by_threshold"]
        summary["settings"]["disagreement_sweep_thresholds"] = disagreement_sweep["settings"]["thresholds"]
        write_json(results_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
