from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path, root=None):
    cfg = read_json(path)
    if root is not None:
        cfg["root"] = str(Path(root).resolve())
    ids = [e["id"] for e in cfg["experiments"]]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("the queue must have eight unique experiments")
    if (cfg["epochs"] != 64 or cfg["segment_epochs"] not in (8, 64) or
            cfg["epochs"] % cfg["segment_epochs"] or cfg["world_size"] != 2):
        raise ValueError("this queue requires 64 epochs, two GPUs, and 8- or 64-epoch jobs")
    for e in cfg["experiments"]:
        if e["batch_size"] * e["accum"] * 2 != 12:
            raise ValueError("effective batch must be 12")
    return cfg


def experiment(cfg, identifier):
    return next(e for e in cfg["experiments"] if e["id"] == identifier)


def run_dir(cfg, identifier):
    return Path(cfg["root"]) / "outputs" / cfg["run_name"] / identifier


def execution_identity(cfg, split):
    root = Path(cfg['root'])
    paths = sorted((root / 'aic/sugon').rglob('*.py'))
    paths += [root / 'aic/eval/map_evaluator.py']
    lock = root / 'configs/sugon_vendors.lock.json'
    if lock.exists():
        paths.append(lock)
    return dict(config_sha256=canonical_hash(cfg), split_sha256=split['sha256'],
                code_sha256=canonical_hash({p.relative_to(root).as_posix(): file_hash(p) for p in paths}))


def find_dataset_roots(root):
    candidates = [p for p in (Path(root) / "dataset").iterdir()
                  if p.is_dir() and all((p / m).is_dir() for m in ("visible", "infrared", "depth"))]
    train = [p for p in candidates if (p / "labels").is_dir()]
    test = [p for p in candidates if not (p / "labels").is_dir()]
    if len(train) != 1 or len(test) != 1:
        raise ValueError("expected exactly one labeled training root and one unlabeled test root")
    return train[0], test[0]


def scan_images(directory):
    found = {}
    for p in sorted(Path(directory).iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            if p.stem in found:
                raise ValueError(f"duplicate image stem: {p.stem}")
            found[p.stem] = p
    return found
