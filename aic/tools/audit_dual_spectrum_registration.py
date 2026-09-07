"""Read-only RGB/IR registration audit using gradient-domain phase correlation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from aic.data.dataset import scan_samples


def load_gray(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def resize_pair(reference: np.ndarray, moving: np.ndarray, max_side: int = 640):
    height, width = reference.shape
    if moving.shape != reference.shape:
        moving = np.asarray(
            Image.fromarray(np.uint8(np.clip(moving, 0, 1) * 255)).resize(
                (width, height), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        shape = (max(32, round(height * scale)), max(32, round(width * scale)))
        reference = ndimage.zoom(reference, (shape[0] / height, shape[1] / width), order=1)
        moving = ndimage.zoom(moving, (shape[0] / height, shape[1] / width), order=1)
    return reference, moving, scale


def gradient(image: np.ndarray) -> np.ndarray:
    image = ndimage.gaussian_filter(image, 1.2)
    gx = ndimage.sobel(image, axis=1)
    gy = ndimage.sobel(image, axis=0)
    value = np.hypot(gx, gy)
    low, high = np.percentile(value, [10, 99])
    return np.clip((value - low) / max(high - low, 1e-6), 0, 1)


def phase_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
    window = np.outer(np.hanning(reference.shape[0]), np.hanning(reference.shape[1]))
    ref_fft = np.fft.fft2((reference - reference.mean()) * window)
    mov_fft = np.fft.fft2((moving - moving.mean()) * window)
    cross = ref_fft * np.conj(mov_fft)
    cross /= np.maximum(np.abs(cross), 1e-8)
    correlation = np.abs(np.fft.ifft2(cross))
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy, dx = float(peak[0]), float(peak[1])
    if dy > reference.shape[0] / 2:
        dy -= reference.shape[0]
    if dx > reference.shape[1] / 2:
        dx -= reference.shape[1]
    return dy, dx


def valid_crop(shape: tuple[int, int], dy: float, dx: float) -> tuple[slice, slice]:
    margin_y = min(max(int(abs(dy)) + 4, 4), shape[0] // 4)
    margin_x = min(max(int(abs(dx)) + 4, 4), shape[1] // 4)
    return slice(margin_y, shape[0] - margin_y), slice(margin_x, shape[1] - margin_x)


def normalized_correlation(a: np.ndarray, b: np.ndarray, crop=None) -> float:
    if crop is not None:
        a, b = a[crop], b[crop]
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))


def mutual_information(a: np.ndarray, b: np.ndarray, crop=None) -> float:
    if crop is not None:
        a, b = a[crop], b[crop]
    histogram, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=32, range=((0, 1), (0, 1)))
    probability = histogram / max(histogram.sum(), 1)
    pa, pb = probability.sum(1), probability.sum(0)
    independent = pa[:, None] * pb[None, :]
    valid = (probability > 0) & (independent > 0)
    return float((probability[valid] * np.log(probability[valid] / independent[valid])).sum())


def edge_overlap(a: np.ndarray, b: np.ndarray, crop=None) -> float:
    edge_a = a >= np.quantile(a, 0.85)
    edge_b = b >= np.quantile(b, 0.85)
    if crop is not None:
        edge_a, edge_b = edge_a[crop], edge_b[crop]
    distance_a = ndimage.distance_transform_edt(~edge_a)
    distance_b = ndimage.distance_transform_edt(~edge_b)
    left = np.exp(-distance_a[edge_b] / 3).mean() if edge_b.any() else 0.0
    right = np.exp(-distance_b[edge_a] / 3).mean() if edge_a.any() else 0.0
    return float((left + right) / 2)


def estimate(reference: np.ndarray, moving: np.ndarray, max_shift: float) -> dict:
    ref_grad, mov_grad = gradient(reference), gradient(moving)
    before_ncc = normalized_correlation(ref_grad, mov_grad)
    before_mi = mutual_information(ref_grad, mov_grad)
    before_edge = edge_overlap(ref_grad, mov_grad)
    best = None
    for angle in (-2.0, -1.0, 0.0, 1.0, 2.0):
        rotated = ndimage.rotate(mov_grad, angle, reshape=False, order=1, mode="constant")
        dy, dx = phase_shift(ref_grad, rotated)
        aligned = ndimage.shift(rotated, (dy, dx), order=1, mode="constant")
        crop = valid_crop(ref_grad.shape, dy, dx)
        ncc = normalized_correlation(ref_grad, aligned, crop)
        mi = mutual_information(ref_grad, aligned, crop)
        edge = edge_overlap(ref_grad, aligned, crop)
        score = (ncc - before_ncc) + 0.25 * (edge - before_edge) + 0.05 * (mi - before_mi)
        candidate = (score, angle, dy, dx, ncc, mi, edge)
        if best is None or candidate > best:
            best = candidate
    score, angle, dy, dx, ncc, mi, edge = best
    magnitude = math.hypot(dx, dy)
    accepted = (
        magnitude <= max_shift
        and ncc >= before_ncc + 0.015
        and edge >= before_edge - 0.01
        and score > 0.02
    )
    return {
        "angle_deg": angle,
        "dy_small": dy,
        "dx_small": dx,
        "shift_small": magnitude,
        "ncc_before": before_ncc,
        "ncc_after": ncc,
        "mi_before": before_mi,
        "mi_after": mi,
        "edge_before": before_edge,
        "edge_after": edge,
        "score": score,
        "accepted": accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_shift_px", type=float, default=64.0)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = scan_samples(args.data_root, require_labels=False)
    rng = random.Random(args.seed)
    selected = rng.sample(samples, min(args.samples, len(samples)))
    rows = []
    for index, sample in enumerate(selected):
        reference, moving, scale = resize_pair(load_gray(sample.visible_path), load_gray(sample.infrared_path))
        row = estimate(reference, moving, max_shift=args.max_shift_px * scale)
        row.update({
            "stem": sample.stem,
            "scale": scale,
            "dx_px": row.pop("dx_small") / scale,
            "dy_px": row.pop("dy_small") / scale,
            "shift_px": row.pop("shift_small") / scale,
        })
        rows.append(row)
        if index % 50 == 0:
            print(f"[registration] {index}/{len(selected)}", flush=True)

    fields = list(rows[0])
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    accepted = [row for row in rows if row["accepted"]]
    summary = {
        "samples": len(rows),
        "accepted": len(accepted),
        "accepted_fraction": len(accepted) / len(rows),
        "median_dx_px": float(np.median([row["dx_px"] for row in accepted])) if accepted else None,
        "median_dy_px": float(np.median([row["dy_px"] for row in accepted])) if accepted else None,
        "median_shift_px": float(np.median([row["shift_px"] for row in accepted])) if accepted else None,
        "p90_shift_px": float(np.quantile([row["shift_px"] for row in accepted], 0.9)) if accepted else None,
        "median_ncc_gain": float(np.median([row["ncc_after"] - row["ncc_before"] for row in accepted])) if accepted else None,
        "median_edge_gain": float(np.median([row["edge_after"] - row["edge_before"] for row in accepted])) if accepted else None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist([row["dx_px"] for row in accepted], bins=30); axes[0].set_title("Accepted dx (px)")
    axes[1].hist([row["dy_px"] for row in accepted], bins=30); axes[1].set_title("Accepted dy (px)")
    axes[2].hist([row["ncc_after"] - row["ncc_before"] for row in rows], bins=30); axes[2].set_title("NCC gain")
    figure.tight_layout()
    figure.savefig(output / "histograms.png", dpi=160)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
