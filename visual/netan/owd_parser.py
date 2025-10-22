# owd_simple_parser.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Union

__all__ = ["SEPARATOR", "OWDRecord", "parse_owd_file", "parse_owd_lines"]

# Change this to ";" or "," if your files use a different delimiter.
SEPARATOR: str = " "


@dataclass(frozen=True)
class OWDRecord:
    """A single OWD observation parsed from a line."""
    timestamp: int
    owd: int
    owd_variation: int


def parse_owd_file(
    path: Union[str, Path],
    *,
    separator: Optional[str] = None,
) -> List[OWDRecord]:
    """
    Parse a file containing space-separated triples:
        <timestamp><sep><owd><sep><owd_variation>

    Only lines with exactly 3 integer-like tokens are accepted.
    Blank lines are ignored. Everything else is skipped.
    """
    sep = SEPARATOR if separator is None else separator
    records: List[OWDRecord] = []

    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for rec in _iter_records(f, sep):
            records.append(rec)

    return records


def parse_owd_lines(
    lines: Iterable[str],
    *,
    separator: Optional[str] = None,
) -> List[OWDRecord]:
    """Parse from an iterable of lines (same rules as `parse_owd_file`)."""
    sep = SEPARATOR if separator is None else separator
    return list(_iter_records(lines, sep))


# --- Internal helpers ---------------------------------------------------------

def _iter_records(lines: Iterable[str], sep: str) -> Iterator[OWDRecord]:
    for raw in lines:
        line = raw.strip()
        if not line:  # ignore blanks
            continue

        parts = line.split(sep)
        if len(parts) != 3:
            continue  # skip non-matching lines

        ts, owd, var = _to_int(parts[0]), _to_int(parts[1]), _to_int(parts[2])
        if ts is None or owd is None or var is None:
            continue  # skip if any token is not an integer

        yield OWDRecord(ts, owd, var)


def _to_int(token: str) -> Optional[int]:
    """
    Convert a token to int if possible; return None if not.
    Accepts leading '+'/'-' signs.
    """
    token = token.strip()
    if not token:
        return None
    try:
        return int(token, 10)
    except ValueError:
        return None
