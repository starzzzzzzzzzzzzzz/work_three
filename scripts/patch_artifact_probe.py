#!/usr/bin/env python3
"""Probe CLIP/ViT patch-score artifacts in LLaVA's vision tower.

This script is intentionally diagnostic-only. It samples a small COCO subset,
extracts LLaVA vision-tower hidden states, computes CLS-patch similarity, maps
COCO boxes onto the 24x24 patch grid, and reports whether high-score patches are
foreground-aligned or background-dominated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_IMAGE_DIR = Path("/Volumes/mydatabase/data/datasets/coco/Image/val2014")
DEFAULT_INSTANCES = Path(
    "/Volumes/mydatabase/data/datasets/coco/Annotations/annotations/instances_val2014.json"
)
DEFAULT_MODEL_PATH = Path("/Volumes/mydatabase/LM/models/llava-1.5-7b-hf")
DEFAULT_OUTPUT_DIR = Path("runs/patch_artifact_probe_50")


@dataclass(frozen=True)
class CocoImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    boxes_xyxy: list[tuple[float, float, float, float]]
    category_names: list[str]
    dominant_area_ratio: float
    num_annotations: int


def normalize_bbox(
    bbox_xywh: Iterable[float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Convert COCO xywh bbox to clipped xyxy coordinates."""
    x, y, w, h = [float(v) for v in bbox_xywh]
    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + max(w, 0.0), 0.0), float(width))
    y2 = min(max(y + max(h, 0.0), 0.0), float(height))
    return x1, y1, x2, y2


def bbox_to_patch_mask(
    boxes_xyxy: Iterable[tuple[float, float, float, float]],
    *,
    width: int,
    height: int,
    grid_size: int,
) -> np.ndarray:
    """Mark patch centers that fall inside any foreground bbox."""
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    if width <= 0 or height <= 0:
        return mask

    boxes = list(boxes_xyxy)
    if not boxes:
        return mask

    cell_w = width / grid_size
    cell_h = height / grid_size
    xs = (np.arange(grid_size, dtype=np.float32) + 0.5) * cell_w
    ys = (np.arange(grid_size, dtype=np.float32) + 0.5) * cell_h

    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        x_hit = (xs >= x1) & (xs <= x2)
        y_hit = (ys >= y1) & (ys <= y2)
        mask |= np.outer(y_hit, x_hit)

    return mask


def compute_patch_artifact_metrics(
    patch_scores: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    topk: int = 50,
) -> dict[str, Any]:
    """Summarize whether high patch-score positions are background-dominated."""
    scores = np.asarray(patch_scores, dtype=np.float32)
    fg = np.asarray(foreground_mask, dtype=bool)
    if scores.shape != fg.shape:
        raise ValueError(f"patch_scores shape {scores.shape} != foreground_mask shape {fg.shape}")

    flat_scores = scores.reshape(-1)
    flat_fg = fg.reshape(-1)
    if flat_scores.size == 0:
        raise ValueError("patch_scores must not be empty")

    top1_idx = int(np.argmax(flat_scores))
    effective_topk = max(1, min(int(topk), flat_scores.size))
    topk_idx = np.argsort(flat_scores)[-effective_topk:][::-1]

    has_fg = bool(flat_fg.any())
    has_bg = bool((~flat_fg).any())
    fg_mean = float(flat_scores[flat_fg].mean()) if has_fg else math.nan
    bg_mean = float(flat_scores[~flat_fg].mean()) if has_bg else math.nan
    fg_bg_gap = fg_mean - bg_mean if has_fg and has_bg else math.nan

    return {
        "point_in_box": float(flat_fg[top1_idx]),
        "top1_is_background": bool(not flat_fg[top1_idx]),
        "top1_patch_index": top1_idx,
        "top1_y": int(top1_idx // scores.shape[1]),
        "top1_x": int(top1_idx % scores.shape[1]),
        "top1_score": float(flat_scores[top1_idx]),
        "topk": effective_topk,
        "topk_background_ratio": float((~flat_fg[topk_idx]).mean()),
        "topk_foreground_ratio": float(flat_fg[topk_idx].mean()),
        "foreground_patch_ratio": float(flat_fg.mean()),
        "num_foreground_patches": int(flat_fg.sum()),
        "fg_score_mean": fg_mean,
        "bg_score_mean": bg_mean,
        "fg_bg_gap": float(fg_bg_gap),
        "score_std": float(flat_scores.std()),
    }


def lowpass_stability_scores_torch(patch_features: Any, *, sigma: float = 0.18) -> Any:
    """Compute channel-frequency low-pass stability scores for patch features.

    Args:
        patch_features: torch tensor with shape [num_patches, hidden_dim].
        sigma: Gaussian width over normalized frequency coordinates.

    Returns:
        torch tensor with shape [num_patches], cosine similarity between original
        and low-pass filtered channel vectors.
    """
    import torch
    import torch.nn.functional as F

    if patch_features.ndim != 2:
        raise ValueError("patch_features must have shape [num_patches, hidden_dim]")
    hidden_dim = patch_features.shape[-1]
    freqs = torch.linspace(-1.0, 1.0, hidden_dim, device=patch_features.device)
    weights = torch.exp(-(freqs**2) / (2 * sigma**2)).to(patch_features.dtype)

    fft_features = torch.fft.fft(patch_features.float(), dim=-1)
    filtered = torch.fft.ifftshift(
        torch.fft.fftshift(fft_features, dim=-1) * weights,
        dim=-1,
    )
    filtered = torch.fft.ifft(filtered, dim=-1).real.to(patch_features.dtype)
    return F.cosine_similarity(patch_features, filtered, dim=-1)


def load_coco_records(
    instances_path: Path,
    image_dir: Path,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    max_annotations: int,
) -> list[CocoImageRecord]:
    """Load COCO image records with usable foreground boxes."""
    with instances_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {int(cat["id"]): cat["name"] for cat in data["categories"]}
    images = {int(img["id"]): img for img in data["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    records: list[CocoImageRecord] = []
    for image_id, anns in anns_by_image.items():
        image = images.get(image_id)
        if image is None:
            continue
        image_path = image_dir / image["file_name"]
        if not image_path.exists():
            continue

        width = int(image["width"])
        height = int(image["height"])
        image_area = max(width * height, 1)

        usable_boxes: list[tuple[float, float, float, float]] = []
        category_names: list[str] = []
        area_ratios: list[float] = []
        for ann in anns:
            x1, y1, x2, y2 = normalize_bbox(ann["bbox"], width=width, height=height)
            area_ratio = max((x2 - x1) * (y2 - y1), 0.0) / image_area
            if area_ratio < min_area_ratio:
                continue
            if area_ratio > max_area_ratio:
                continue
            usable_boxes.append((x1, y1, x2, y2))
            category_names.append(categories.get(int(ann["category_id"]), str(ann["category_id"])))
            area_ratios.append(area_ratio)

        if not usable_boxes:
            continue
        if len(usable_boxes) > max_annotations:
            continue

        records.append(
            CocoImageRecord(
                image_id=image_id,
                file_name=image["file_name"],
                width=width,
                height=height,
                boxes_xyxy=usable_boxes,
                category_names=category_names,
                dominant_area_ratio=float(max(area_ratios)),
                num_annotations=len(usable_boxes),
            )
        )

    records.sort(key=lambda r: (r.num_annotations, abs(r.dominant_area_ratio - 0.25), r.image_id))
    return records


def sample_records(
    records: list[CocoImageRecord],
    *,
    num_images: int,
    seed: int,
) -> list[CocoImageRecord]:
    """Sample from clearer records while keeping deterministic randomness."""
    rng = random.Random(seed)
    single = [r for r in records if r.num_annotations == 1]
    pool = single if len(single) >= num_images else records
    pool = pool[: max(num_images * 20, num_images)]
    selected = rng.sample(pool, k=min(num_images, len(pool)))
    selected.sort(key=lambda r: r.image_id)
    return selected


def _choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_vision_components(model_path: Path, device: str):
    import torch
    from safetensors import safe_open
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under {model_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors.index.json under {model_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    vision_config = CLIPVisionConfig(**config["vision_config"])
    feature_layer = int(config.get("vision_feature_layer", -2))

    processor = CLIPImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    vision_model = CLIPVisionModel(vision_config)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    vision_keys = [key for key in weight_map if key.startswith("vision_tower.")]
    state_dict = {}
    for shard_name in sorted({weight_map[key] for key in vision_keys}):
        shard_path = model_path / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in vision_keys:
                if weight_map[key] == shard_name:
                    state_dict[key.removeprefix("vision_tower.")] = shard.get_tensor(key)

    missing, unexpected = vision_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Vision tower load mismatch. missing={missing}, unexpected={unexpected}")

    dtype = torch.float16 if device == "cuda" else torch.float32
    vision_model.to(device=device, dtype=dtype)
    vision_model.eval()
    return processor, vision_model, feature_layer


def _extract_scores_for_image(
    processor: Any,
    vision_model: Any,
    image_path: Path,
    device: str,
    *,
    feature_layer: int,
) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    dtype = next(vision_model.parameters()).dtype
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)

    with torch.inference_mode():
        outputs = vision_model(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[feature_layer][0]
        cls = hidden[0]
        patches = hidden[1:]
        patch_scores = F.cosine_similarity(patches, cls.unsqueeze(0), dim=-1)
        # Run FFT-based stability on CPU for macOS/MPS compatibility.
        stability_scores = lowpass_stability_scores_torch(patches.float().cpu())

    grid_size = int(round(math.sqrt(patches.shape[0])))
    if grid_size * grid_size != patches.shape[0]:
        raise RuntimeError(f"Expected square patch grid, got {patches.shape[0]} patches")

    return {
        "patch_scores": patch_scores.float().cpu().numpy().reshape(grid_size, grid_size),
        "stability_scores": stability_scores.float().cpu().numpy().reshape(grid_size, grid_size),
    }


def write_overlay_png(
    image_path: Path,
    output_path: Path,
    *,
    patch_scores: np.ndarray,
    foreground_mask: np.ndarray,
) -> None:
    """Save a lightweight overlay for manual inspection."""
    import matplotlib.pyplot as plt
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    score = patch_scores.astype(np.float32)
    score = (score - score.min()) / max(float(score.max() - score.min()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[1].imshow(image)
    axes[1].imshow(score, cmap="magma", alpha=0.55, extent=(0, image.width, image.height, 0))
    axes[1].set_title("PatchScore")
    axes[2].imshow(image)
    axes[2].imshow(foreground_mask.astype(float), cmap="Greens", alpha=0.45, extent=(0, image.width, image.height, 0))
    axes[2].set_title("COCO foreground")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_probe(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    records = load_coco_records(
        Path(args.instances),
        Path(args.image_dir),
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        max_annotations=args.max_annotations,
    )
    selected = sample_records(records, num_images=args.num_images, seed=args.seed)
    if not selected:
        raise RuntimeError("No usable COCO records found. Relax filters or check paths.")

    device = _choose_device(args.device)
    processor, vision_model, feature_layer = _load_vision_components(Path(args.model_path), device)
    grid_size = vision_model.config.image_size // vision_model.config.patch_size

    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(selected, start=1):
        image_path = Path(args.image_dir) / record.file_name
        extracted = _extract_scores_for_image(
            processor,
            vision_model,
            image_path,
            device,
            feature_layer=feature_layer,
        )
        patch_scores = extracted["patch_scores"]
        stability_scores = extracted["stability_scores"]
        foreground_mask = bbox_to_patch_mask(
            record.boxes_xyxy,
            width=record.width,
            height=record.height,
            grid_size=grid_size,
        )

        metrics = compute_patch_artifact_metrics(patch_scores, foreground_mask, topk=args.topk)
        stability_metrics = compute_patch_artifact_metrics(
            stability_scores,
            foreground_mask,
            topk=args.topk,
        )

        row = {
            "rank_input_order": idx,
            "image_id": record.image_id,
            "file_name": record.file_name,
            "width": record.width,
            "height": record.height,
            "categories": "|".join(record.category_names),
            "num_annotations": record.num_annotations,
            "dominant_area_ratio": record.dominant_area_ratio,
            **{f"patch_{k}": v for k, v in metrics.items()},
            **{f"stability_{k}": v for k, v in stability_metrics.items()},
        }
        rows.append(row)

        np.savez_compressed(
            arrays_dir / f"{record.image_id:012d}.npz",
            patch_scores=patch_scores,
            stability_scores=stability_scores,
            foreground_mask=foreground_mask,
        )
        if args.save_overlays:
            write_overlay_png(
                image_path,
                overlay_dir / f"{record.image_id:012d}_{record.file_name}.png",
                patch_scores=patch_scores,
                foreground_mask=foreground_mask,
            )
        print(
            f"[{idx:03d}/{len(selected):03d}] image_id={record.image_id} "
            f"top1_bg={metrics['top1_is_background']} "
            f"topk_bg={metrics['topk_background_ratio']:.3f} "
            f"gap={metrics['fg_bg_gap']:.4f}"
        )

    rows.sort(
        key=lambda r: (
            float(r["patch_top1_is_background"]),
            float(r["patch_topk_background_ratio"]),
            -float(r["patch_fg_bg_gap"]) if not math.isnan(float(r["patch_fg_bg_gap"])) else -999.0,
        ),
        reverse=True,
    )

    csv_path = output_dir / "patch_artifact_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_images": len(rows),
        "model_path": str(args.model_path),
        "image_dir": str(args.image_dir),
        "instances": str(args.instances),
        "grid_size": grid_size,
        "top1_background_rate": float(np.mean([r["patch_top1_is_background"] for r in rows])),
        "mean_topk_background_ratio": float(np.mean([r["patch_topk_background_ratio"] for r in rows])),
        "mean_fg_bg_gap": float(np.nanmean([r["patch_fg_bg_gap"] for r in rows])),
        "mean_stability_topk_background_ratio": float(
            np.mean([r["stability_topk_background_ratio"] for r in rows])
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--min-area-ratio", type=float, default=0.05)
    parser.add_argument("--max-area-ratio", type=float, default=0.65)
    parser.add_argument("--max-annotations", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--save-overlays", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
