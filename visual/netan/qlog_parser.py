"""
qlog_parser
-------------
A minimal, library-style parser for QUIC qlog files (mvfst-style) that extracts
per-timestamp congestion-control and transport metrics.
It simply parses and normalizes what's present in the qlog.

It builds a "timeline" of samples keyed by event timestamp; each sample may
contain any of the following raw fields when available in the log:

 - cwnd (current_cwnd), ssthresh
 - bytes_in_flight
 - congestion_event, recovery_state, congestion_state ("state")
 - min_rtt, latest_rtt (a.k.a. raw RTT), smoothed_rtt, ack_delay
 - bandwidth_estimate (as a raw pair: bandwidth_bytes, bandwidth_interval)
 - bytes_sent (from packet_sent), bytes_lost (from packets_lost)
 - bytes_acked (incremental at the event that carried the ACK)
 - sent_bytes, lost_bytes, acked_bytes (cumulative running totals)
 - acked_ranges / acked_packet_count (if present in ACK frames)
 - congestion control algorithm name (cc_algo), if hinted by transport_state_update

The parser is tolerant to both array-based qlog "events" and
object-shaped events, and to multiple traces in a single file.

Example
-------
from qlog_parser import QlogParser
result = QlogParser().parse_file("path/to/file.qlog")
len(result.timeline)
12345
sample = result.timeline[0]
sample.cwnd, sample.min_rtt

Notes
-----
- This parser now computes **bytes_acked** using ACK frames in `packet_received`
 (supports both `ack` and `ack_receive_timestamps` variants) by matching
 acked packet numbers to previously observed `packet_sent` sizes.
- It also exposes convenient cumulative counters: `sent_bytes`, `lost_bytes`,
 and `acked_bytes` on every sample. Incremental values per event remain in
 `bytes_sent`, `bytes_lost`, and `bytes_acked`.
- Units are preserved as-is from the source log (timestamps are strings in qlog;
 this parser normalizes them to integers when possible).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# -----------------------------
# Data models
# -----------------------------

@dataclass
class TimelineSample:
    """A unified per-timestamp snapshot of raw metrics present in the qlog.
    All fields are optional; only what's present at that timestamp is set.
    """
    ts: int  # qlog timestamp as integer if possible; otherwise best-effort parsed
    category: Optional[str] = None
    name: Optional[str] = None

    # Congestion metrics (metric_update / congestion_metric_update)
    cwnd: Optional[int] = None              # current_cwnd
    ssthresh: Optional[int] = None
    bytes_in_flight: Optional[int] = None
    congestion_event: Optional[str] = None
    recovery_state: Optional[str] = None
    congestion_state: Optional[str] = None  # "state" in some logs

    # RTT metrics (recovery:metric_update)
    min_rtt: Optional[int] = None
    latest_rtt: Optional[int] = None
    smoothed_rtt: Optional[int] = None
    ack_delay: Optional[int] = None

    # Bandwidth estimate (bandwidth_est_update)
    bandwidth_bytes: Optional[int] = None
    bandwidth_interval: Optional[int] = None

    # Transport / packets (incremental at this timestamp)
    bytes_sent: Optional[int] = None         # from packet_sent.header.packet_size
    bytes_lost: Optional[int] = None         # from packets_lost.lost_bytes
    bytes_acked: Optional[int] = None        # derived at packet_received with ACK frames

    # Cumulative running totals up to this timestamp
    sent_bytes: Optional[int] = None
    lost_bytes: Optional[int] = None
    acked_bytes: Optional[int] = None

    # Loss details
    lost_packets: Optional[int] = None       # from packets_lost.lost_packets
    largest_lost_packet_num: Optional[int] = None

    # ACKs (raw)
    acked_ranges: Optional[List[Tuple[int, int]]] = None  # if present in ack frames
    acked_packet_count: Optional[int] = None              # if present

    # Misc helpful
    packet_type: Optional[str] = None
    packet_number: Optional[int] = None

    # Congestion control algorithm (if hinted at this timestamp)
    cc_algo: Optional[str] = None

    # Catch-all raw payload for debugging / future-proofing
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QlogTraceInfo:
    title: Optional[str] = None
    qlog_version: Optional[str] = None
    vantage_point: Optional[Dict[str, Any]] = None
    common_fields: Optional[Dict[str, Any]] = None

@dataclass
class QlogParseResult:
    traces: List[QlogTraceInfo]
    timeline: List[TimelineSample]

def to_dict(self) -> Dict[str, Any]:
    return {
        "traces": [asdict(t) for t in self.traces],
        "timeline": [asdict(s) for s in self.timeline],
    }

# -----------------------------
# Internal helpers
# -----------------------------

def _to_int(value: Union[str, int]) -> int:
    """Best-effort conversion of qlog timestamps (often strings) to int."""
    try:
        return int(value)
    except (ValueError, TypeError):
        # As a last resort, drop non-digits and try again
        if isinstance(value, str):
            digits = ''.join(ch for ch in value if ch.isdigit())
            if digits:
                return int(digits)
        raise

def _ensure_list(obj: Any) -> List[Any]:
    return obj if isinstance(obj, list) else [obj]

def _extract_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Safe dict traversal using dotted path (e.g., 'header.packet_size')."""
    cur = d
    for key in path.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

# -----------------------------
# Main parser
# -----------------------------

class QlogParser:
    """Parses mvfst-style qlog files into a unified timeline of raw metrics."""

    # Known event names we care about (mvfst variants)
    _EV_CONG_METRIC = "congestion_metric_update"
    _EV_RECOVERY_METRIC = "metric_update"           # usually category 'recovery'
    _EV_BW_EST = "bandwidth_est_update"
    _EV_PKT_SENT = "packet_sent"
    _EV_PKT_RECV = "packet_received"
    _EV_PKTS_LOST = "packets_lost"
    _EV_TRANSPORT_STATE = "transport_state_update"

    def parse_file(self, path: Union[str, Path]) -> QlogParseResult:
        """Parse a qlog file and return trace info + a per-event timeline.
        - Handles ACK frames under both `ack` and `ack_receive_timestamps` (and hyphenated),
            extracting `acked_ranges` and deriving `bytes_acked` using packet sizes from
            earlier `packet_sent` events.
        - Provides incremental `bytes_sent` / `bytes_lost` and cumulative
            `sent_bytes` / `lost_bytes` / `acked_bytes` on every sample.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            root = json.load(f)

        traces_info: List[QlogTraceInfo] = []
        timeline: List[TimelineSample] = []

        # Collect basic trace info
        for tr in _ensure_list(root.get("traces", [])):
            traces_info.append(
                QlogTraceInfo(
                    title=root.get("title"),
                    qlog_version=root.get("qlog_version"),
                    vantage_point=tr.get("vantage_point"),
                    common_fields=tr.get("common_fields"),
                )
            )

        # --- State for cumulative accounting and ACK attribution ---
        cumulative_sent = 0
        cumulative_lost = 0
        cumulative_acked = 0

        # Map packet_number -> size (track sent sizes for later ack credit)
        # We avoid double counting the same PN across multiple ACK frames.
        sent_packet_sizes: Dict[int, int] = {}
        acked_pns_seen: set = set()

        # Parse events across all traces in-order
        for tr in _ensure_list(root.get("traces", [])):
            events = tr.get("events") or []
            for ev in events:
                ts, category, name, data = self._normalize_event(ev)
                if data is None:
                    continue  # skip malformed

                sample = TimelineSample(ts=ts, category=category, name=name, raw=data.copy())

                # Congestion metrics (transport/metric_update variant)
                if name == self._EV_CONG_METRIC:
                    sample.cwnd = data.get("current_cwnd")
                    sample.ssthresh = data.get("ssthresh")
                    sample.bytes_in_flight = data.get("bytes_in_flight")
                    sample.congestion_event = data.get("congestion_event")
                    sample.recovery_state = data.get("recovery_state")
                    sample.congestion_state = data.get("state")

                # Recovery RTT metrics
                elif name == self._EV_RECOVERY_METRIC and category == "recovery":
                    sample.min_rtt = data.get("min_rtt")
                    sample.latest_rtt = data.get("latest_rtt")
                    sample.smoothed_rtt = data.get("smoothed_rtt")
                    sample.ack_delay = data.get("ack_delay")

                # Bandwidth estimate updates
                elif name == self._EV_BW_EST:
                    sample.bandwidth_bytes = data.get("bandwidth_bytes")
                    sample.bandwidth_interval = data.get("bandwidth_interval")

                # Packet sent
                elif name == self._EV_PKT_SENT:
                    pkt_size = _extract_nested(data, "header.packet_size") or 0
                    pkt_num = _extract_nested(data, "header.packet_number")
                    sample.bytes_sent = pkt_size
                    sample.packet_number = pkt_num
                    sample.packet_type = data.get("packet_type")
                    if isinstance(pkt_size, int) and pkt_size > 0:
                        cumulative_sent += pkt_size
                    if isinstance(pkt_num, int) and isinstance(pkt_size, int):
                        sent_packet_sizes[pkt_num] = pkt_size

                # Packet received (may contain ACK frames)
                elif name == self._EV_PKT_RECV:
                    sample.packet_number = _extract_nested(data, "header.packet_number")
                    sample.packet_type = data.get("packet_type")
                    frames = data.get("frames") or []
                    ack_ranges: List[Tuple[int, int]] = []
                    ack_count: Optional[int] = None
                    for fr in frames:
                        ft = str(fr.get("frame_type", "")).lower()
                        if ft in ("ack", "ack_frame", "acknowledgement",
                                    "ack_receive_timestamps", "ack-receive-timestamps"):
                            rngs = fr.get("acked_ranges")
                            if isinstance(rngs, list):
                                for r in rngs:
                                    if isinstance(r, (list, tuple)) and len(r) == 2:
                                        try:
                                            lo, hi = int(r[0]), int(r[1])
                                            if hi < lo:
                                                lo, hi = hi, lo
                                            ack_ranges.append((lo, hi))
                                        except Exception:
                                            pass
                            if "acked_packet_count" in fr:
                                try:
                                    ack_count = int(fr.get("acked_packet_count"))
                                except Exception:
                                    pass
                    if ack_ranges:
                        sample.acked_ranges = ack_ranges

                        # Derive incremental bytes_acked for this event using our sent map
                        inc_acked = 0
                        for lo, hi in ack_ranges:
                            # inclusive range
                            for pn in range(lo, hi + 1):
                                if pn in sent_packet_sizes and pn not in acked_pns_seen:
                                    inc_acked += sent_packet_sizes[pn]
                                    acked_pns_seen.add(pn)
                        if inc_acked:
                            sample.bytes_acked = inc_acked
                            cumulative_acked += inc_acked

                    if ack_count is not None:
                        sample.acked_packet_count = ack_count

                # Packets lost
                elif name == self._EV_PKTS_LOST:
                    lost_bytes = data.get("lost_bytes")
                    if isinstance(lost_bytes, int) and lost_bytes > 0:
                        sample.bytes_lost = lost_bytes
                        cumulative_lost += lost_bytes
                    sample.lost_packets = data.get("lost_packets")
                    sample.largest_lost_packet_num = data.get("largest_lost_packet_num")

                # Transport state updates (e.g., "CCA set to westwood")
                elif name == self._EV_TRANSPORT_STATE:
                    upd = str(data.get("update", "")).lower()
                    # Heuristic: extract CC algo name if present
                    # Examples: "CCA set to westwood", "CCA set to bbr", etc.
                    if "cca set to" in upd:
                        algo = upd.split("cca set to", 1)[-1].strip()
                        sample.cc_algo = algo

                # Add cumulative running totals to every sample for convenience
                sample.sent_bytes = cumulative_sent
                sample.lost_bytes = cumulative_lost
                sample.acked_bytes = cumulative_acked

                # Append the sample for this exact timestamp; caller can post-process
                timeline.append(sample)
        return QlogParseResult(traces=traces_info, timeline=timeline)

    # ----------
    # Utilities
    # ----------

    def _normalize_event(self, ev: Union[List[Any], Dict[str, Any]]) -> Tuple[int, str, str, Optional[Dict[str, Any]]]:
        """Normalize an event to (ts, category, name, data) tuple.

        Supports mvfst array form: [ts, category, name, data]
        and object form: {"time": ..., "category": ..., "name": ..., "data": {...}}
        """
        if isinstance(ev, list):
            if len(ev) < 4:
                return (0, "", "", None)
            ts_raw, category, name, data = ev[0], ev[1], ev[2], ev[3]
            try:
                ts = _to_int(ts_raw)
            except Exception:
                ts = 0
            if not isinstance(data, dict):
                data = {}
            return (ts, str(category), str(name), data)

        if isinstance(ev, dict):
            ts_raw = ev.get("time") or ev.get("ts") or ev.get("timestamp")
            category = ev.get("category", "")
            name = ev.get("name", "")
            data = ev.get("data") or ev.get("event_data") or {}
            try:
                ts = _to_int(ts_raw) if ts_raw is not None else 0
            except Exception:
                ts = 0
            if not isinstance(data, dict):
                data = {}
            return (ts, str(category), str(name), data)

        return (0, "", "", None)

__all__ = ["QlogParser", "QlogParseResult", "TimelineSample", "QlogTraceInfo"]
