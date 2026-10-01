"""Deterministic, training-only object/context crops of real source images.

No image duplication, synthetic pixels, new depth measurements or validation
labels are used. The ordinary image sampler still visits each source once.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import math
import random

import numpy as np


def view_seed(seed, epoch, stem, purpose):
    text = f"{seed}/{epoch}/{stem}/{purpose}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


@contextmanager
def isolated_random(seed):
    """Do not let crop decisions change worker/collate or model RNG streams."""
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def crop_boxes(boxes, source_hw, window):
    """Return all clipped boxes, intersection flags and original area fractions."""
    h, w = source_hw
    top, left, ch, cw = window
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    xy1 = (boxes[:, :2] - boxes[:, 2:] / 2) * [w, h]
    xy2 = (boxes[:, :2] + boxes[:, 2:] / 2) * [w, h]
    a = np.maximum(xy1, [left, top])
    b = np.minimum(xy2, [left + cw, top + ch])
    wh = np.maximum(b - a, 0)
    visible = np.all(wh > 0, axis=1)
    fraction = wh.prod(1) / np.maximum((xy2 - xy1).prod(1), 1e-12)
    result = np.concatenate(((a + b) / 2 - [left, top], wh), axis=1)
    result /= [cw, ch, cw, ch]
    # Completely outside boxes are discarded by the caller, not relabelled.
    return result.astype(np.float32), visible, fraction


class ContextViewPolicy:
    def __init__(self, samples, config, size_hw, min_box_px=2.0):
        self.config = dict(config)
        self.enabled = bool(config.get("enabled", False))
        self.seed = int(config.get("seed", 42))
        self.size_hw = tuple(size_hw)
        self.min_box_px = float(min_box_px)
        self.max_short = float(config.get("anchor_max_short_px", 64))
        self.target_range = tuple(config.get("target_short_px", [32, 64]))
        self.zoom_range = tuple(config.get("zoom_range", [1.5, 3.0]))
        self.min_visible = float(config.get("min_visible_fraction", 0.5))
        self.min_context = float(config.get("min_context_ratio", 2.0))
        self.attempts = int(config.get("attempts", 24))
        self.stop_epoch = int(config.get("stop_epoch", 12))
        if not (0 < self.target_range[0] <= self.target_range[1]
                and 1 < self.zoom_range[0] <= self.zoom_range[1]
                and 0 < self.min_visible <= 1 and self.min_context > 1
                and self.attempts > 0):
            raise ValueError("invalid context-view geometry")
        stems = [s.stem for s in samples]
        if len(set(stems)) != len(stems):
            raise ValueError("context views require unique source identities")
        # Two complementary halves: each source retains ordinary geometry every
        # other epoch, even when every proposed crop is accepted.
        ordered = sorted(stems, key=lambda s: view_seed(self.seed, 0, s, "cohort"))
        self.cohort = {s: i % 2 for i, s in enumerate(ordered)}
        counts = Counter(c for s in samples for c in s.class_ids)
        median = float(np.median(list(counts.values()))) if counts else 1.0
        self.class_weights = {c: min(2.0, max(1.0, math.sqrt(median / n)))
                              for c, n in counts.items()}

    def choose(self, stem, epoch, boxes, labels, source_hw):
        h, w = source_hw
        meta = {"requested": False, "used": False, "reason": "ordinary",
                "anchor": -1, "anchor_class": -1, "zoom": 1.0,
                "source_hw": [h, w], "window": [0, 0, h, w]}
        if not self.enabled or epoch >= self.stop_epoch or self.cohort[stem] != epoch % 2:
            return None, meta
        meta["requested"] = True
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        short = np.min(boxes[:, 2:] * [self.size_hw[1], self.size_hw[0]], axis=1)
        eligible = np.flatnonzero((short > 0) & (short < self.max_short))
        if not len(eligible):
            meta["reason"] = "no_small_anchor"
            return None, meta
        rng = random.Random(view_seed(self.seed, epoch, stem, "geometry"))
        weights = [self.class_weights.get(int(labels[i]), 1.0) * min(2., max(1., 32 / short[i]))
                   for i in eligible]
        anchor = int(rng.choices(eligible.tolist(), weights=weights, k=1)[0])
        meta.update(anchor=anchor, anchor_class=int(labels[anchor]), source_short=float(short[anchor]))
        cx, cy, bw, bh = boxes[anchor] * [w, h, w, h]
        for _ in range(self.attempts):
            desired = rng.uniform(*self.target_range)
            zoom = min(self.zoom_range[1], max(self.zoom_range[0], desired / short[anchor]))
            ch, cw = max(1, round(h / zoom)), max(1, round(w / zoom))
            # Context must fit around the anchor in both dimensions.
            if cw < self.min_context * bw or ch < self.min_context * bh:
                continue
            left_lo = max(0, math.ceil(cx + bw / 2 - cw))
            left_hi = min(w - cw, math.floor(cx - bw / 2))
            top_lo = max(0, math.ceil(cy + bh / 2 - ch))
            top_hi = min(h - ch, math.floor(cy - bh / 2))
            if left_lo > left_hi or top_lo > top_hi:
                continue
            left, top = rng.randint(left_lo, left_hi), rng.randint(top_lo, top_hi)
            window = (top, left, ch, cw)
            clipped, visible, fraction = crop_boxes(boxes, (h, w), window)
            px_wh = clipped[:, 2:] * [self.size_hw[1], self.size_hw[0]]
            # No ignore regions exist in the criterion. Reject the whole view
            # when an intersecting neighbour would become an unlabelled sliver.
            if np.any(visible & ((fraction < self.min_visible) | np.any(px_wh < self.min_box_px, axis=1))):
                continue
            if not visible[anchor] or fraction[anchor] < 0.999:
                continue
            meta.update(used=True, reason="context", window=list(window), zoom=float(w / cw),
                        retained=int(visible.sum()), clipped=int(np.sum(visible & (fraction < .999))),
                        anchor_short=float(np.min(px_wh[anchor])))
            return window, meta
        meta["reason"] = "unsafe_boundary_or_context"
        return None, meta
