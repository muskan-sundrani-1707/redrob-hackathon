"""Load and iterate candidate records from JSONL."""

import json
from pathlib import Path
from typing import Iterator, List, Optional, Union


def load_candidates(path: Union[str, Path]) -> Iterator[dict]:
    """Stream candidates from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates_list(path: Union[str, Path], limit: Optional[int] = None) -> List[dict]:
    """Load candidates into a list (optionally capped)."""
    candidates = []
    for i, c in enumerate(load_candidates(path)):
        candidates.append(c)
        if limit and i + 1 >= limit:
            break
    return candidates
