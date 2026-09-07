#!/usr/bin/env python3
"""Build a SWA-style checkpoint by averaging saved training checkpoints.

The script prefers EMA weights when present. The output checkpoint stores the
averaged state under both ``ema`` and ``model`` so existing inference code can
load it without special handling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def list_epoch_checkpoints(save_dir: Path) -> list[Path]:
    pairs: list[tuple[int, Path]] = []
    for path in save_dir.glob("epoch*.pt"):
        stem = path.stem
        if not stem.startswith("epoch"):
            continue
        try:
            epoch = int(stem[5:])
        except ValueError:
            continue
        pairs.append((epoch, path))
    pairs.sort(key=lambda item: item[0])
    return [path for _, path in pairs]


def pick_state(ckpt: object) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(ckpt, dict):
        ema = ckpt.get("ema")
        if isinstance(ema, dict) and ema:
            return ema, "ema"
        model = ckpt.get("model")
        if isinstance(model, dict) and model:
            return model, "model"
    if isinstance(ckpt, dict):
        tensors = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
        if tensors:
            return tensors, "raw"
    raise TypeError("checkpoint does not contain a tensor state dict")


def average_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("no states to average")
    ref = states[0]
    avg: dict[str, torch.Tensor] = {}
    for key, value in ref.items():
        if torch.is_tensor(value):
            avg[key] = value.detach().clone().float() if value.is_floating_point() else value.detach().clone()
        else:
            avg[key] = value

    for state in states[1:]:
        if state.keys() != ref.keys():
            missing = sorted(ref.keys() - state.keys())
            extra = sorted(state.keys() - ref.keys())
            raise ValueError(f"state dict keys differ; missing={missing[:5]} extra={extra[:5]}")
        for key, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point():
                avg[key].add_(value.detach().float())
    count = float(len(states))
    for key, value in avg.items():
        if torch.is_tensor(value) and value.is_floating_point():
            avg[key].div_(count)
    return avg


def resolve_epochs(
    checkpoints: list[Path],
    *,
    last_k: int | None = None,
    epochs: list[int] | None = None,
) -> list[Path]:
    if epochs:
        by_epoch = {}
        for path in checkpoints:
            try:
                by_epoch[int(path.stem[5:])] = path
            except ValueError:
                continue
        selected = []
        for epoch in epochs:
            if epoch not in by_epoch:
                raise FileNotFoundError(f"missing checkpoint epoch{epoch}.pt")
            selected.append(by_epoch[epoch])
        return selected
    if last_k is None or last_k <= 0:
        raise ValueError("last_k must be positive when epochs are not provided")
    if len(checkpoints) < last_k:
        raise ValueError(f"need at least {last_k} checkpoints, found {len(checkpoints)}")
    return checkpoints[-last_k:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True, help="directory containing epoch*.pt")
    parser.add_argument("--output", required=True, help="path of the averaged checkpoint")
    parser.add_argument("--last_k", type=int, default=6, help="average the latest K checkpoints")
    parser.add_argument(
        "--epochs",
        default=None,
        help="optional comma-separated list of epochs to average; overrides --last_k",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    output = Path(args.output)
    checkpoints = list_epoch_checkpoints(save_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no epoch*.pt checkpoints in {save_dir}")

    selected_epochs = [int(x) for x in args.epochs.split(",")] if args.epochs else None
    selected = resolve_epochs(checkpoints, last_k=args.last_k, epochs=selected_epochs)

    states = []
    kinds = []
    for path in selected:
        ckpt = torch.load(path, map_location="cpu")
        state, kind = pick_state(ckpt)
        states.append(state)
        kinds.append(kind)

    averaged = average_states(states)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "ema": averaged,
        "model": averaged,
        "swa_epochs": [int(p.stem[5:]) for p in selected],
        "swa_checkpoints": [p.name for p in selected],
        "swa_source_kinds": kinds,
        "source": "build_swa_checkpoint.py",
    }
    torch.save(meta, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "epochs": meta["swa_epochs"],
                "source_kinds": kinds,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
