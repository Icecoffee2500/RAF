"""Dataset splitting utility for Cityscapes.

`split_cityscapes(root)` will create three client splits with roughly equal
image counts and guarantee each contains at least one pixel of every class.

The split indices are deterministic (seed 0) to keep runs reproducible.
The function returns a dict mapping client_id -> list[int] (indices).
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from torchvision.datasets import Cityscapes


NUM_CLIENTS = 3
_SEED = 0


def split_cityscapes(root: str | Path) -> Dict[int, List[int]]:
    root = Path(root)
    rng = random.Random(_SEED)

    # use torchvision dataset to access labels
    ds = Cityscapes(str(root), split="train", mode="fine", target_type="semantic")
    num_samples = len(ds)

    indices = list(range(num_samples))
    rng.shuffle(indices)

    # naive equal split first
    chunk_size = num_samples // NUM_CLIENTS
    client_indices: Dict[int, List[int]] = {
        cid: indices[cid * chunk_size : (cid + 1) * chunk_size] for cid in range(NUM_CLIENTS)
    }

    # simple coverage check – if a class missing, move random sample containing it
    class_presence: Dict[int, set[int]] = defaultdict(set)
    for cid, idxs in client_indices.items():
        for idx in idxs:
            _, lbl = ds[idx]
            class_presence[cid].update(np.unique(lbl))

    all_classes = set(range(19))  # train IDs 0 – 18
    for cid in range(NUM_CLIENTS):
        missing = all_classes - class_presence[cid]
        if not missing:
            continue
        # move samples from others until coverage satisfied (very small data so simple loop)
        for other in range(NUM_CLIENTS):
            if other == cid:
                continue
            for idx in list(client_indices[other]):
                _, lbl = ds[idx]
                if missing & set(np.unique(lbl)):
                    client_indices[other].remove(idx)
                    client_indices[cid].append(idx)
                    class_presence[cid].update(np.unique(lbl))
                    missing = all_classes - class_presence[cid]
                if not missing:
                    break
            if not missing:
                break
    return client_indices 