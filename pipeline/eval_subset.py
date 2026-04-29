# optional khaya/claude budget cap: if data/synthetic/evaluation_subset_ids.txt exists,
# only those narrative ids are processed by roundtrip, tts, pipeline batch, extraction-on-roundtrip.
# if the file is missing, all rows are processed (legacy 126).
# one integer id per line; lines starting with # are ignored.
from __future__ import annotations

import functools
from pathlib import Path

SUBSET_PATH = Path("data/synthetic/evaluation_subset_ids.txt")


@functools.lru_cache(maxsize=1)
def get_eval_id_set() -> frozenset[int] | None:
    if not SUBSET_PATH.exists():
        return None
    text = SUBSET_PATH.read_text(encoding="utf-8")
    ids: set[int] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.add(int(line))
    return frozenset(ids) if ids else None


def use_eval_subset() -> bool:
    return get_eval_id_set() is not None


def id_in_eval_subset(row_id: int) -> bool:
    s = get_eval_id_set()
    if s is None:
        return True
    return int(row_id) in s
