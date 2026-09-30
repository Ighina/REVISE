"""Sharded on-disk store for anchor activations (float16), keyed by string."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class ActivationStore:
    def __init__(self, root: Path, shard_size: int = 256):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self.shard_size = shard_size
        self._index: Dict[str, Tuple[int, int]] = {}
        self._n_shards = 0
        if self.index_path.exists():
            with open(self.index_path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self._index[r["key"]] = (r["shard"], r["row"])
                        self._n_shards = max(self._n_shards, r["shard"] + 1)
        self._buf_keys: List[str] = []
        self._buf: List[np.ndarray] = []
        self._cache: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ write
    def __contains__(self, key: str) -> bool:
        return key in self._index or key in self._buf_keys

    def keys(self) -> List[str]:
        return list(self._index) + list(self._buf_keys)

    def append(self, keys: Sequence[str], arr: np.ndarray) -> None:
        assert arr.shape[0] == len(keys)
        for k, row in zip(keys, arr):
            if k in self:
                continue
            self._buf_keys.append(k)
            self._buf.append(np.asarray(row, dtype=np.float16))
            if len(self._buf) >= self.shard_size:
                self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        shard = self._n_shards
        np.save(self.root / f"shard_{shard:05d}.npy", np.stack(self._buf, axis=0))
        with open(self.index_path, "a") as f:
            for i, k in enumerate(self._buf_keys):
                self._index[k] = (shard, i)
                f.write(json.dumps({"key": k, "shard": shard, "row": i}) + "\n")
        self._n_shards += 1
        self._buf_keys, self._buf = [], []

    # ------------------------------------------------------------------ read
    def _shard(self, s: int) -> np.ndarray:
        if s not in self._cache:
            self._cache[s] = np.load(self.root / f"shard_{s:05d}.npy", mmap_mode="r")
        return self._cache[s]

    def get(self, key: str, layers: Optional[Sequence[int]] = None) -> np.ndarray:
        s, r = self._index[key]
        a = self._shard(s)[r]
        return np.asarray(a[list(layers)] if layers is not None else a)

    def get_many(self, keys: Iterable[str], layers: Optional[Sequence[int]] = None) -> np.ndarray:
        keys = list(keys)
        if not keys:
            return np.zeros((0,), dtype=np.float16)
        return np.stack([self.get(k, layers) for k in keys], axis=0)

    @property
    def shape_tail(self) -> Tuple[int, ...]:
        if not self._index:
            return ()
        s, _ = next(iter(self._index.values()))
        return tuple(self._shard(s).shape[1:])

    def __len__(self) -> int:
        return len(self._index) + len(self._buf_keys)
