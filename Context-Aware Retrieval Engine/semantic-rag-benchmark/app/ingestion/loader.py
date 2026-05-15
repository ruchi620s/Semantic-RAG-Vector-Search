from __future__ import annotations

from pathlib import Path
from typing import List


def load_technical_paragraphs(file_path: str) -> List[str]:
    """Loads technical paragraphs separated by blank lines."""

    content = Path(file_path).read_text(encoding="utf-8")
    parts = [block.strip() for block in content.split("\n\n")]
    return [part for part in parts if part]
