from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np

from aic.data.modalities import (
    DEPTH_KIND_BAD,
    DEPTH_KIND_DISP8,
    DEPTH_KIND_MM16,
    load_depth_raw,
    preprocess_depth,
)


def scan(root: str, parse_only: bool = False) -> dict:
    depth_dir = os.path.join(root, "depth")
    counts: Counter[str] = Counter()
    quality_means: list[float] = []
    valid_means: list[float] = []
    for name in sorted(os.listdir(depth_dir)):
        path = os.path.join(depth_dir, name)
        if not os.path.isfile(path):
            continue
        raw, kind, sensor_quality = load_depth_raw(path)
        counts[kind] += 1
        if parse_only:
            continue
        _, valid, quality = preprocess_depth(raw, kind, sensor_quality)
        if kind != DEPTH_KIND_BAD:
            valid_means.append(float(valid.mean()))
            quality_means.append(float(quality.mean()))
        if not (
            raw.ndim == valid.ndim == quality.ndim == 2
            and raw.shape == valid.shape == quality.shape
            and np.isfinite(quality).all()
            and quality.min(initial=0.0) >= 0.0
            and quality.max(initial=0.0) <= 1.0
        ):
            raise RuntimeError(f"invalid depth-v2 tensors: {path}")
    return {
        "total": sum(counts.values()),
        "mm16": counts[DEPTH_KIND_MM16],
        "disp8": counts[DEPTH_KIND_DISP8],
        "bad": counts[DEPTH_KIND_BAD],
        "mean_valid": float(np.mean(valid_means)) if valid_means else None,
        "mean_quality": float(np.mean(quality_means)) if quality_means else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()
    for root in args.roots:
        print(root, scan(root, parse_only=args.parse_only))


if __name__ == "__main__":
    main()
