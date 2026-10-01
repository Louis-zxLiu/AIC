"""Label-only, deterministic grouped stratification; never examines model scores."""
from __future__ import annotations

import argparse
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import atomic_json, canonical_hash, file_hash, read_json, scan_images

SEQUENCE = re.compile(r"(?:hehe_\d+_)?(\d{6})_(\d{3})(?:_suppl)?_(\d{8})")


def group_key(stem):
    match = SEQUENCE.fullmatch(stem)
    return f"sequence:{match[1]}_{match[2]}" if match else f"image:{stem}"


def source_key(stem):
    if stem.startswith("hehe_"):
        return 0
    if stem.startswith("shuming_"):
        return 1
    return 2 if SEQUENCE.fullmatch(stem) else 3


def label_record(path):
    counts = np.zeros(12, dtype=np.int64)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid label row in {path}")
        cls = int(fields[0])
        coords = [float(v) for v in fields[1:]]
        # Original annotations can extend slightly beyond the image. Count them
        # unchanged here; the shared loader clips XYXY before train/evaluation.
        if not 0 <= cls < 12 or not all(math.isfinite(v) for v in coords) or min(coords[2:]) <= 0:
            raise ValueError(f"invalid label values in {path}")
        counts[cls] += 1
    return counts


def build_split(label_dir, seed=42, candidates=10000):
    files = sorted(Path(label_dir).glob("*.txt"))
    if len(files) != 2000:
        raise ValueError(f"expected 2000 labels, found {len(files)}")
    members = defaultdict(list)
    vectors = {}
    hashes = {}
    for p in files:
        key = group_key(p.stem)
        counts = label_record(p)
        vector = np.zeros(29, dtype=np.int64)
        vector[0], vector[1:13], vector[13:25] = 1, counts, counts > 0
        vector[25 + source_key(p.stem)] = 1
        vectors[key] = vectors.get(key, np.zeros(29, dtype=np.int64)) + vector
        members[key].append(p.stem)
        hashes[p.stem] = file_hash(p)
    keys = sorted(members)
    rows = np.stack([vectors[k] for k in keys])
    total = rows.sum(axis=0)
    low = np.ceil(total * .15).astype(int)
    high = np.floor(total * .25).astype(int)
    low[0] = high[0] = 400
    low[12], high[12], low[24], high[24] = 5, 7, 4, 7
    # Source proportions are objective terms, not additional feasibility gates.
    low[25:], high[25:] = 0, total[25:]
    rng = random.Random(seed)
    best = None
    for _ in range(candidates):
        order = list(range(len(keys)))
        rng.shuffle(order)
        selected, size = [], 0
        for j in order:
            n = int(rows[j, 0])
            if size + n <= 400:
                selected.append(j)
                size += n
            if size == 400:
                break
        counts = rows[selected].sum(axis=0)
        if np.any(counts < low) or np.any(counts > high):
            continue
        objective = float(np.abs(counts[1:] - total[1:] * .2).dot(1 / np.maximum(total[1:] * .2, 1)))
        val = tuple(sorted(stem for j in selected for stem in members[keys[j]]))
        score = (objective, val)
        if best is None or score < best[0]:
            best = (score, counts)
    if best is None:
        raise RuntimeError("no feasible grouped stratified split; do not silently relax constraints")
    val = list(best[0][1])
    val_set = set(val)
    result = {
        "version": 1, "seed": seed, "candidates": candidates,
        "train": sorted(p.stem for p in files if p.stem not in val_set), "val": val,
        "groups": {k: members[k] for k in keys}, "label_sha256": hashes,
        "statistics": {"total_instances": total[1:13].tolist(), "val_instances": best[1][1:13].tolist(),
                       "total_images_with_class": total[13:25].tolist(), "val_images_with_class": best[1][13:25].tolist(),
                       "source_names": ["hehe", "shuming", "recognized_sequence", "numeric_other"],
                       "total_sources": total[25:].tolist(), "val_sources": best[1][25:].tolist()},
        "grouping_limit": "Only filename-identifiable sequences are grouped; other image identities do not guarantee scene independence.",
    }
    result["sha256"] = canonical_hash(result)
    validate_split(result)
    return result


def validate_split(split, data_root=None):
    body = {k: v for k, v in split.items() if k != "sha256"}
    if canonical_hash(body) != split.get("sha256"):
        raise ValueError("split content hash mismatch")
    train, val = set(split["train"]), set(split["val"])
    if len(train) != 1600 or len(val) != 400 or train & val or len(split["train"]) != 1600 or len(split["val"]) != 400:
        raise ValueError("invalid train/val partition")
    all_members = [s for values in split["groups"].values() for s in values]
    if len(all_members) != 2000 or set(all_members) != train | val:
        raise ValueError("group members must cover the dataset exactly")
    for values in split["groups"].values():
        if set(values) & train and set(values) & val:
            raise ValueError("a recognized sequence crosses the split")
    if data_root is not None:
        for modality in ("visible", "infrared", "depth"):
            if set(scan_images(Path(data_root) / modality)) != train | val:
                raise ValueError(f"{modality} image identities do not match the frozen split")
        for stem, digest in split["label_sha256"].items():
            if file_hash(Path(data_root) / "labels" / f"{stem}.txt") != digest:
                raise ValueError(f"label changed after split freeze: {stem}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates", type=int, default=10000)
    args = parser.parse_args()
    result = build_split(args.labels, candidates=args.candidates)
    dest = Path(args.output)
    if dest.exists() and read_json(dest) != result:
        raise RuntimeError("refusing to replace an existing different frozen split")
    atomic_json(dest, result)
    print(result["sha256"], result["statistics"], flush=True)


if __name__ == "__main__":
    main()
