# queue_series.py
"""
Lightweight parser for Vegvisir queue logs that produces a clean time series of
queue filling (backlog) over time for the shared bottleneck on eth0:netem.

This module is dependency-free (standard library only) and intended to be used
inside a larger logging/analysis library.

Example
-------
from queue_series import parse_bottleneck_series

series = parse_bottleneck_series(
    run_dir="/path/to/.../shaper",   # directory containing queue_total.csv
    device="eth0",                   # fixed egress where all flows aggregate (this topology)
    qdisc="netem"                    # fixed leaf qdisc 

print(series.device, series.qdisc)
print(series.t_s[:5], series.backlog_bytes[:5])
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class QueueSeries:
    """
    A simple, analysis-friendly structure holding the queue filling series.
    - t_ns: monotonic timestamp in nanoseconds (relative to the logger start)
    - t_s:  seconds since the first sample in this series (float)
    - backlog_bytes: queue occupancy in bytes (sum over the selected qdisc)
    - backlog_pkts:  queue occupancy in packets (optional signal)
    - meta: metadata about the selection performed (device, qdisc)
    """
    device: str
    qdisc: str
    t_ns: List[int]
    t_s: List[float]
    backlog_bytes: List[int]
    backlog_pkts: List[int]
    meta: Dict[str, str]


def _read_queue_total(path: Path) -> List[dict]:
    """
    Read queue_total.csv into a list of dict rows (typed where possible).
    Expected header:
      t_ns,service,device,qdisc,backlog_bytes,backlog_pkts,
      drops_total,overlimits_total,requeues_total,ecn_mark_total
    """
    rows: List[dict] = []
    if path.is_dir():
        path = path / "queue_total.csv"
    if not path.exists():
        raise FileNotFoundError(f"queue_total.csv not found at: {path}")

    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            # Defensive: coerce numeric fields; skip rows missing critical fields
            try:
                t_ns = int((r.get("t_ns") or "").strip())
                device = (r.get("device") or "").strip()
                qdisc = (r.get("qdisc") or "").strip()
                if not device or not qdisc:
                    continue
                backlog_bytes = int((r.get("backlog_bytes") or "0").strip())
                backlog_pkts  = int((r.get("backlog_pkts") or "0").strip())
            except Exception:
                continue

            rows.append({
                "t_ns": t_ns,
                "device": device,
                "qdisc": qdisc,
                "backlog_bytes": backlog_bytes,
                "backlog_pkts": backlog_pkts,
            })
    # Ensure deterministic order by time, then device/qdisc
    rows.sort(key=lambda d: (d["t_ns"], d["device"], d["qdisc"]))
    return rows


def _aggregate_by_time(
    rows: Iterable[dict],
    device: str,
    qdisc: str
) -> Tuple[List[int], List[int], List[int]]:
    """
    Reduce rows to a single series for (device, qdisc):
      - group by t_ns and sum backlog_bytes/packets
      - sort by t_ns
    Returns (t_ns, backlog_bytes, backlog_pkts)
    """
    buckets: Dict[int, Tuple[int, int]] = {}  # t_ns -> (bytes_sum, pkts_sum)
    for r in rows:
        if r["device"] != device or r["qdisc"] != qdisc:
            continue
        t_ns = r["t_ns"]
        b = r["backlog_bytes"]
        p = r["backlog_pkts"]
        if t_ns in buckets:
            prev_b, prev_p = buckets[t_ns]
            buckets[t_ns] = (prev_b + b, prev_p + p)
        else:
            buckets[t_ns] = (b, p)

    if not buckets:
        return [], [], []

    t_seq = sorted(buckets.keys())
    bytes_seq = [buckets[t][0] for t in t_seq]
    pkts_seq  = [buckets[t][1] for t in t_seq]
    return t_seq, bytes_seq, pkts_seq


def parse_bottleneck_series(
    run_dir: str | Path,
    device: str = "eth0",
    qdisc: str = "netem",
) -> QueueSeries:
    """
    parser for the shared bottleneck on eth0:netem (no heuristics).

    - `device` defaults to "eth0" (egress towards clients in the given topology).
    - `qdisc`  defaults to "netem" (leaf where queueing delay is realized).

    Raises:
      FileNotFoundError if queue_total.csv cannot be found.
      ValueError if device/qdisc cannot be located in the file.
    """
    run_dir = Path(run_dir)
    rows = _read_queue_total(run_dir / "queue_total.csv")

    t_ns, backlog_bytes, backlog_pkts = _aggregate_by_time(rows, device, qdisc)
    if not t_ns:
        # Provide a helpful diagnostic listing what *was* present
        present = sorted({(r["device"], r["qdisc"]) for r in rows})
        raise ValueError(
            f"No samples for device='{device}', qdisc='{qdisc}'. "
            f"Pairs present in file: {present}"
        )

    t0 = t_ns[0]
    t_s = [(t - t0) / 1e9 for t in t_ns]

    return QueueSeries(
        device=device,
        qdisc=qdisc,
        t_ns=t_ns,
        t_s=t_s,
        backlog_bytes=backlog_bytes,
        backlog_pkts=backlog_pkts,
        meta={
            "source": str((run_dir / "queue_total.csv").resolve()),
            "topology_note": "clients-eth0-sim-eth1-server; using egress leaf netem on eth0"
        },
    )


# Optional: small CLI for quick inspection
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Extract eth0:netem bottleneck queue series from Vegvisir logs.")
    ap.add_argument("--run-dir", required=True, help="Path to shaper directory (contains queue_total.csv)")
    ap.add_argument("--device", default="eth0", help="Device to parse (default: eth0)")
    ap.add_argument("--qdisc", default="netem", help="Qdisc to select (default: netem)")
    ap.add_argument("--head", type=int, default=5, help="Print first N samples (default: 5)")
    args = ap.parse_args()

    series = parse_bottleneck_series(args.run_dir, device=args.device, qdisc=args.qdisc)
    print(json.dumps({
        "device": series.device,
        "qdisc": series.qdisc,
        "samples": min(args.head, len(series.t_ns)),
        "t_ns": series.t_ns[:args.head],
        "t_s": series.t_s[:args.head],
        "backlog_bytes": series.backlog_bytes[:args.head],
        "backlog_pkts": series.backlog_pkts[:args.head],
        "meta": series.meta,
    }, indent=2))
