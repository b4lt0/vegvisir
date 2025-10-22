#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Local imports (tools)
import sys
from .qlog_parser import QlogParser
from .queue_parser import parse_bottleneck_series  
from .owd_parser import parse_owd_file, OWDRecord  


@dataclass
class FlowInput:
    qlog_path: Path
    owd_path: Path
    flow_id: Optional[str] = None


@dataclass
class FlowOutput:
    flow_id: str
    timeline: List[Dict[str, Any]]


@dataclass
class QueueSeriesOut:
    device: str
    qdisc: Optional[str]
    t_ns: List[int]
    t_s: List[float]
    backlog_bytes: List[int]
    backlog_pkts: List[int]
    meta: Dict[str, Any]


@dataclass
class OWDSeriesOut:
    t_ms: List[int]                 # timestamps in ms
    owd_us: List[int]               # OWD in microseconds
    owd_variation: Optional[List[int]] = None
    meta: Optional[Dict[str, Any]] = None



@dataclass
class RunOutput:
    flows: List[FlowOutput]
    queue_series: Optional[QueueSeriesOut] = None
    owd_series: Optional[OWDSeriesOut] = None

    


def _tiny_parse_owd_file(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Fallback OWD reader: expects 3 tokens per line: ts  owd_us  delta_us (delta may be negative)."""
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            # Accept 2 or 3 columns; treat 2nd as owd_us, 3rd optional as variation
            try:
                ts = int(parts[0])
                owd_us = int(parts[1])
                owd_var = int(parts[2]) if len(parts) > 2 else None
            except ValueError:
                continue
            rec = {"ts": ts, "owd_us": owd_us}
            if owd_var is not None:
                rec["owd_var_us"] = owd_var
            recs.append(rec)
    return recs


def _to_dict_list(objs: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for o in objs:
        if hasattr(o, "__dict__"):
            d = dict(o.__dict__)
        else:
            d = dict(o)
        out.append(d)
    return out

class RunProcessor:
    def __init__(self, queue_run_dir: Union[str, Path], device: str = "eth0", qdisc: Optional[str] = "netem"):
        self.queue_run_dir = Path(queue_run_dir)
        self.device = device
        self.qdisc = qdisc
        self._qlog_paths: List[Path] = []
        self._owd_path: Optional[Path] = None

    def add_qlog(self, path: Path) -> None:
        self._qlog_paths.append(Path(path))
        if not Path(path).exists():
            raise FileNotFoundError(f"qlog not found: {path}")

    def set_owd_file(self, path: Path) -> None:
        self._owd_path = Path(path)
        if not Path(path).exists():
            raise FileNotFoundError(f"owd not found: {path}")

    def _infer_flow_id(self, parsed: Any, fallback_path: Path, explicit: Optional[str] = None) -> str:
        if explicit:
            return explicit
        # Try DCID from common_fields
        try:
            traces = getattr(parsed, "traces", None) or []
            if traces:
                cf = traces[0].common_fields or {}
                dcid = cf.get("dcid") if isinstance(cf, dict) else None
                if dcid:
                    return str(dcid)
        except Exception:
            pass
        # Fallback: stem of qlog filename
        return fallback_path.stem

    def _owd_records_to_series(self, recs):
        t_ms = []
        owd_us = []
        owd_var = []
        for r in recs:
            # r likely has .timestamp (ms), .owd (us), and maybe .owd_variation (us)
            t_ms.append(r.timestamp)
            owd_us.append(r.owd)
            owd_var.append(getattr(r, "owd_variation", None))
        # If variation is entirely None, set to None; else keep the list
        if all(v is None for v in owd_var):
            owd_var = None
        return OWDSeriesOut(t_ms=t_ms, owd_us=owd_us, owd_variation=owd_var, meta={"source": str(self._owd_path)})


    def run(self) -> RunOutput:
        # Parse queue
        if parse_bottleneck_series is None:
            raise RuntimeError("queue_parser.parse_bottleneck_series not available")
        series = parse_bottleneck_series(self.queue_run_dir, device=self.device, qdisc=(self.qdisc or "netem"))
        queue_out = QueueSeriesOut(
            device=getattr(series, "device", self.device),
            qdisc=getattr(series, "qdisc", self.qdisc or "netem"),
            t_ns=list(getattr(series, "t_ns", []) or []),
            t_s=list(getattr(series, "t_s", []) or []),
            backlog_bytes=list(getattr(series, "backlog_bytes", []) or []),
            backlog_pkts=list(getattr(series, "backlog_pkts", []) or []),
            meta=dict(getattr(series, "meta", {}) or {}),
        )

        flows_out: List[FlowOutput] = []
        for qpath in self._qlog_paths:
            if QlogParser is None:
                raise RuntimeError("qlog_parser.QlogParser not available")
            qp = QlogParser()
            parsed = qp.parse_file(str(qpath))
            flow_id = self._infer_flow_id(parsed, qpath)#, fi.flow_id)
            timeline = [asdict(s) for s in parsed.timeline]
            flows_out.append(FlowOutput(flow_id=flow_id, timeline=timeline))

            owd_records = []
            if self._owd_path is not None:
                owd_records = parse_owd_file(self._owd_path)
            owd_out = self._owd_records_to_series(owd_records)


        return RunOutput(flows=flows_out, queue_series=queue_out, owd_series=owd_out)


# ------------------------------- CLI --------------------------------------- #

def dataclass_or_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [dataclass_or_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: dataclass_or_dict(v) for k, v in obj.items()}
    return obj

def main():
    ap = argparse.ArgumentParser(description="Process multiple (qlog, owd) flows and a queue series.")
    ap.add_argument("--qlog", action="append", default=[], help="Path to a qlog file. Repeat per flow.")
    ap.add_argument("--owd", action="append", default=[], help="Path to an OWD file. Repeat per flow.")
    ap.add_argument("--queue-run-dir", required=True, help="Directory containing queue logs (e.g., queue_total.csv).")
    ap.add_argument("--device", default="eth0")
    ap.add_argument("--qdisc", default="netem")
    ap.add_argument("--out-json", default=None, help="Optional: path to write combined JSON payload.")
    ap.add_argument("--print-summary", action="store_true", help="Print a short summary to stdout.")
    args = ap.parse_args()

    if not args.qlog or not args.owd or len(args.qlog) != len(args.owd):
        raise SystemExit("Provide matching pairs with --qlog ... --owd ... (repeat once per flow).")

    rp = RunProcessor(queue_run_dir=args.queue_run_dir, device=args.device, qdisc=args.qdisc)
    for q, o in zip(args.qlog, args.owd):
        rp.add_pair(q, o)
    run_output = rp.run()

    if args.print_summary:
        print(f"Flows: {len(run_output.flows)}")
        for f in run_output.flows:
            print(f"- {f.flow_id}: timeline={len(f.timeline)} owd_samples={len(f.owd_records)}")
        dev = run_output.queue_series.device
        qdisc = run_output.queue_series.qdisc or args.qdisc
        print(f"Queue: device={dev} qdisc={qdisc} samples={len(run_output.queue_series.t_s)}")

    if args.out_json:
        payload = dataclass_or_dict(run_output)
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
