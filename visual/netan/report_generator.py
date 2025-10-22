# netan/report_generator.py
from pathlib import Path
from datetime import datetime
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class ReportGenerator:
    """
    Minimal reporter:
      - Inputs: run_output (as produced by RunProcessor.run()), out_dir (str|Path)
      - Outputs: per-flow PDFs, one multi-flow PDF, and 'report.csv'
      - Returns: path to 'report.csv'
    """

    def __init__(self, run_output, out_dir, show_p50=True, show_p90=True, showfliers=False):
        self.ro = run_output
        self.out_dir = Path(out_dir)
        self.show_p50 = show_p50
        self.show_p90 = show_p90
        self.showfliers = showfliers

    # ---------------- public API ----------------
    def generate(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._save_per_flow_pdfs()
        self._save_multiflow_pdf()
        csv_path = self._write_report_csv()
        self._write_report_txt()
        return str(csv_path)

    def print_report(self, stream=sys.stdout):
        """Print a concise, human-readable report (no over-engineering)."""
        flows = getattr(self.ro, "flows", []) or []
        q = getattr(self.ro, "queue_series", None)

        # Per-flow metrics (reuse same calc as CSV)
        rows = [self._per_flow_metrics(f) for f in flows]

        # Queue meta + stats over full series
        dev = getattr(q, "device", "eth0") if q is not None else "eth0"
        qdisc = getattr(q, "qdisc", "netem") if q is not None else "netem"
        qbytes = (getattr(q, "backlog_bytes", []) or []) if q is not None else []
        if qbytes:
            q_avg, q_std = self._mean_std(qbytes)
        else:
            q_avg = q_std = ""

        print("=== Congestion Control Metrics Report ===", file=stream)
        print(f"Flows: {len(rows)}", file=stream)
        print(f"Queue: device={dev} qdisc={qdisc}", file=stream)
        if q_avg != "":
            print(f"Queue backlog (avg ± std):  {self._fmt_bytes(int(q_avg))} ± {self._fmt_bytes(int(q_std))}", file=stream)
        print("", file=stream)

        # Per-flow pretty section
        for f, r in zip(flows, rows):
            cc_algo = self._first_cc_algo(f)
            flow_id = r["flow_id"]
            print(f"- Flow: {flow_id}  (cc={cc_algo})", file=stream)
            print(f"  Duration:                 {r['duration_s']:10.3f} s", file=stream)
            print(f"  Throughput:               {r['throughput_Mbps']:10.3f} Mbps", file=stream)
            print(f"  Goodput:                  {r['goodput_Mbps']:10.3f} Mbps", file=stream)
            print(f"  Loss rate:                {r['loss_rate_pct']:5.2f} %", file=stream)

            # OWD / RTT lines only if we have values
            if r["owd_avg_ms"] != "":
                print(f"  OWD avg ± std:            {r['owd_avg_ms']:10.3f} ± {r['owd_std_ms']:10.3f} ms", file=stream)
            if r["rtt_latest_avg_ms"] != "":
                rtt_min = (f"{r['rtt_min_ms']:.3f}" if r["rtt_min_ms"] != "" else "n/a")
                print(f"  RTT avg ± std (min):      {r['rtt_latest_avg_ms']:10.3f} ± {r['rtt_latest_std_ms']:10.3f} ms (min={rtt_min} ms)", file=stream)
            print("", file=stream)

        # Multi-flow summary
        tot_thr = sum(r["throughput_Mbps"] for r in rows if isinstance(r["throughput_Mbps"], (int,float)))
        tot_good = sum(r["goodput_Mbps"] for r in rows if isinstance(r["goodput_Mbps"], (int,float)))
        print("=== Multi-Flow Summary ===", file=stream)
        print(f"Total Throughput:           {tot_thr:10.3f} Mbps", file=stream)
        print(f"Total Goodput:              {tot_good:10.3f} Mbps", file=stream)
        print(f"Jain Fairness (Throughput): {self._jain([r['throughput_Mbps'] for r in rows]):.4f}", file=stream)
        print(f"Jain Fairness (Goodput):    {self._jain([r['goodput_Mbps']    for r in rows]):.4f}", file=stream)

    # ---------------- small helpers ----------------
    def _get_run_owd_series(self):
        return getattr(self.ro, "owd_series", None)

    def _get_run_owd_arrays_ms(self):
        """
        Returns (t_s, owd_ms) from run-level OWD series.
        t_s is seconds since first OWD timestamp.
        owd_ms are OWD values in milliseconds.
        """
        s = self._get_run_owd_series()
        if not s or not getattr(s, "t_ms", None):
            return [], []
        t0 = s.t_ms[0]
        t_s = [(t - t0) / 1000.0 for t in s.t_ms]   # ms -> s
        owd_ms = [v / 1000.0 for v in s.owd_us]     # µs -> ms
        return t_s, owd_ms


    def _compute_bytes_acked_inline(self, flow):
        """
        Return a new timeline list where events that *received* ACK frames
        (acknowledging *our* packets) have an incremental 'bytes_acked' field.

        - Tracks sent packet sizes by (packet_type, packet_number) to respect PN spaces.
        - Never double-counts: once a PN is credited, it won't be credited again.
        - Does not overwrite an existing nonzero 'bytes_acked'.
        """
        tl = getattr(flow, "timeline", []) or []
        if not tl:
            return tl

        # 1) Collect sent packet sizes by PN space
        sent_sizes = {}            # key: (space, pn) -> size
        for r in tl:
            try:
                bs = r.get("bytes_sent")
                pn = r.get("packet_number")
                if bs is None or pn is None:
                    continue
                space = r.get("packet_type") or "1RTT"
                key = (str(space), int(pn))
                # if the same PN shows up multiple times, keep the max size seen
                prev = sent_sizes.get(key, 0)
                cur = int(bs)
                if cur > prev:
                    sent_sizes[key] = cur
            except Exception:
                # keep parsing robust
                pass

        # 2) Walk ACK-bearing *received* events and assign incremental bytes_acked
        acked_keys = set()         # which (space, pn) we've already credited
        out = []
        for r in tl:
            # copy-on-write to avoid mutating the original object list
            rr = dict(r)

            if rr.get("bytes_acked"):
                out.append(rr)
                continue

            ack_ranges = rr.get("acked_ranges")
            if not ack_ranges:
                out.append(rr)
                continue

            # Only count when these ACKs are *from peer to us*: packet_received
            name = (rr.get("name") or rr.get("event") or "").lower()
            if name not in ("packet_received",) and not name.endswith("packet_received"):
                out.append(rr)
                continue

            space = rr.get("packet_type") or "1RTT"
            newly_acked = 0
            try:
                for rng in ack_ranges:
                    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                        continue
                    lo, hi = int(rng[0]), int(rng[1])
                    # mvfst-style qlogs typically use inclusive ranges
                    for pn in range(lo, hi + 1):
                        key = (str(space), pn)
                        if key in acked_keys:
                            continue
                        sz = sent_sizes.get(key)
                        if sz:
                            newly_acked += int(sz)
                        acked_keys.add(key)
            except Exception:
                # be conservative; if anything goes wrong we just don't add bytes_acked here
                newly_acked = 0

            if newly_acked > 0:
                rr["bytes_acked"] = newly_acked

            out.append(rr)

        return out

    
    @staticmethod
    def _fmt_bytes(n: int) -> str:
        # human-readable, binary units
        units = ["B","KiB","MiB","GiB","TiB"]
        n = float(n)
        i = 0
        while n >= 1024.0 and i < len(units)-1:
            n /= 1024.0
            i += 1
        return f"{n:.1f} {units[i]}"

    @staticmethod
    def _first_cc_algo(flow):
        tl = getattr(flow, "timeline", []) or []
        for r in tl:
            cca = r.get("cc_algo")
            if cca:
                return str(cca)
        return "n/a"

    @staticmethod
    def _jain(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        if not xs:
            return 0.0
        num = (sum(xs)) ** 2
        den = len(xs) * sum(x*x for x in xs)
        return (num / den) if den > 0 else 0.0
    @staticmethod
    def _mean_std(arr):
        if not arr:
            return ("", "")
        a = np.array(arr, dtype=float)
        return (float(a.mean()), float(a.std(ddof=0)))

    def _duration_seconds(self, flow):
        tl = getattr(flow, "timeline", []) or []
        ts = [r["ts"] for r in tl if r.get("ts") is not None]
        if ts:
            return (max(ts) - min(ts)) / 1_000_000.0  # µs -> s
        s = self._get_run_owd_series()
        if s and getattr(s, "t_ms", None):
            return (max(s.t_ms) - min(s.t_ms)) / 1000.0  # ms -> s
        return 0.0

    # ---------------- figures (return fig or None) ----------------
    def _fig_owd(self, flow):
        t_s, owd_ms = self._get_run_owd_arrays_ms()
        if not t_s:
            return None
        fig, ax = plt.subplots()
        ax.plot(t_s, owd_ms, linewidth=1.2)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("OWD (ms)")
        ax.set_title("One-Way-Delay")
        fig.tight_layout()
        return fig

    def _fig_owd_queue(self, flow, q):
        if q is None:
            return None
        owd_t_s, owd_ms = self._get_run_owd_arrays_ms()
        if not owd_t_s:
            return None


        tns = getattr(q, "t_ns", []) or []
        qbytes = getattr(q, "backlog_bytes", []) or []
        if not tns or not qbytes:
            return None
        qt0 = tns[0]
        q_t_s = [(t - qt0) / 1e9 for t in tns]

        odur = owd_t_s[-1]
        q_idx_end = len(q_t_s) - 1
        while q_idx_end > 0 and q_t_s[q_idx_end] > odur:
            q_idx_end -= 1
        q_t = q_t_s[:q_idx_end + 1]
        q_mb = [b / 1_000_000.0 for b in qbytes[:q_idx_end + 1]]

        fig, ax1 = plt.subplots()
        ax1.plot(owd_t_s, owd_ms, linewidth=1.2)
        ax1.set_xlabel("Time from OWD start (s)")
        ax1.set_ylabel("OWD (ms)")
        ax2 = ax1.twinx()
        if q_t:
            ax2.plot(q_t, q_mb, linewidth=1.0)
        ax2.set_ylabel("Queue backlog (MB)")
        fid = (getattr(flow, "flow_id", "flow") or "flow")[:8]
        dev = getattr(q, "device", ""); qdisc = getattr(q, "qdisc", "")
        ax1.set_title(f"OWD vs Queue ({dev}:{qdisc})")
        ax1.set_xlim(0.0, odur)
        fig.tight_layout()
        return fig

    def _fig_cwnd_ssthresh_events(self, flow):
        tl = getattr(flow, "timeline", []) or []
        if not tl:
            return None
        cwnd_t, cwnd_b, ssth_t, ssth_b, ev_t = [], [], [], [], []
        for r in tl:
            ts = r.get("ts")
            if ts is None:
                continue
            c = r.get("cwnd")
            if c is not None:
                cwnd_t.append(ts); cwnd_b.append(c)
            s = r.get("ssthresh")
            if s is not None:
                ssth_t.append(ts); ssth_b.append(0 if s < 0 else s)
            if r.get("congestion_event") == "congestion packet loss":
                ev_t.append(ts)

        CAP = 62_500  # 500 Kb -> 62,500 bytes
        cwnd = [(t, v) for t, v in zip(cwnd_t, cwnd_b) if v <= CAP]
        ssth = [(t, v) for t, v in zip(ssth_t, ssth_b) if v <= CAP]
        if not (cwnd or ssth):
            return None

        t0 = min([t for t, _ in (cwnd + ssth)] + (ev_t or [cwnd[0][0] if cwnd else ssth[0][0]]))
        c_t = [(t - t0) / 1e6 for t, _ in cwnd]; c_kB = [v / 1000.0 for _, v in cwnd]
        s_t = [(t - t0) / 1e6 for t, _ in ssth]; s_kB = [v / 1000.0 for _, v in ssth]
        e_t = [(t - t0) / 1e6 for t in ev_t]

        fig, ax = plt.subplots()
        if c_t: ax.plot(c_t, c_kB, label="cwnd (KB)", linewidth=1.2)
        if s_t: ax.plot(s_t, s_kB, label="ssthresh (KB)", linewidth=1.2, linestyle="--")
        if e_t:
            yl = ax.get_ylim()
            ax.vlines(e_t, yl[0], yl[1], linewidth=0.6, alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("KB")
        ax.set_title(f"Flow {getattr(flow,'flow_id','flow')[:8]} — cwnd/ssthresh + loss")
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        return fig

    def _fig_rtt_latest_min(self, flow):
        tl = getattr(flow, "timeline", []) or []
        if not tl:
            return None
        tL, rL, tM, rM = [], [], [], []
        for r in tl:
            ts = r.get("ts")
            if ts is None:
                continue
            lr = r.get("latest_rtt"); mr = r.get("min_rtt")
            if lr is not None:
                tL.append(ts); rL.append(lr)
            if mr is not None:
                tM.append(ts); rM.append(mr)
        if not (tL or tM):
            return None
        t0 = min((tL or [tM[0]]) + (tM or []))
        tL = [(t - t0) / 1e6 for t in tL]; rL = [v / 1000.0 for v in rL]
        tM = [(t - t0) / 1e6 for t in tM]; rM = [v / 1000.0 for v in rM]
        fig, ax = plt.subplots()
        if tL: ax.plot(tL, rL, label="latest RTT (ms)", linewidth=1.2)
        if tM: ax.plot(tM, rM, label="min RTT (ms)", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("RTT (ms)")
        ax.set_title(f"Flow {getattr(flow,'flow_id','flow')[:8]} — RTT latest & min")
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        return fig

    def _fig_bytes_cum(self, flow):
        tl = self._compute_bytes_acked_inline(getattr(flow, "timeline", []) or [])
        tl = [r for r in tl if r.get("ts") is not None]
        if not tl:
            return None
        tl.sort(key=lambda r: r["ts"])
        t0 = tl[0]["ts"]
        t, sMB, aMB, lMB = [], [], [], []
        cs = ca = cl = 0
        for r in tl:
            upd = False
            bs = r.get("bytes_sent")
            if bs is not None: cs += bs; upd = True
            ba = r.get("bytes_acked")
            if ba is not None: ca += ba; upd = True
            bl = r.get("bytes_lost")
            if bl is not None: cl += bl; upd = True
            if upd:
                t.append((r["ts"] - t0) / 1e6)
                sMB.append(cs / 1_000_000.0); aMB.append(ca / 1_000_000.0); lMB.append(cl / 1_000_000.0)
        if not t:
            return None
        fig, ax = plt.subplots()
        ax.plot(t, sMB, label="sent (MB)", linewidth=1.2)
        ax.plot(t, aMB, label="acked (MB)", linewidth=1.2)
        ax.plot(t, lMB, label="lost (MB)", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Cumulative MB")
        ax.set_title(f"Flow {getattr(flow,'flow_id','flow')[:8]} — bytes cumulative")
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        return fig

    def _fig_owd_all(self):
        flows = getattr(self.ro, "flows", []) or []
        if not flows:
            return None
        fig, ax = plt.subplots()
        t_s, owd_ms = self._get_run_owd_arrays_ms()
        if not t_s:
            return None
        ax.plot(t_s, owd_ms, linewidth=1.0)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("OWD (ms)")
        ax.set_title("One-Way-Delay")
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        return fig

    def _fig_ecdf_latest_all(self):
        flows = getattr(self.ro, "flows", []) or []
        if not flows:
            return None
        fig, ax = plt.subplots()
        anyp = False
        for f in flows:
            tl = getattr(f, "timeline", []) or []
            x = np.array([r["latest_rtt"] / 1000.0 for r in tl if r.get("latest_rtt") is not None], dtype=float)
            if x.size == 0:
                continue
            x.sort()
            u, cnt = np.unique(x, return_counts=True)
            cdf = np.cumsum(cnt) / x.size
            ax.step(u, cdf, where="post", linewidth=1.2, label=(getattr(f, "flow_id", "flow") or "flow")[:8])
            if self.show_p50:
                ax.vlines([np.percentile(x, 50)], 0, 1, linewidth=0.6, alpha=0.2)
            if self.show_p90:
                ax.vlines([np.percentile(x, 90)], 0, 1, linewidth=0.6, alpha=0.2)
            anyp = True
        if not anyp:
            return None
        ax.set_xlabel("latest RTT (ms)"); ax.set_ylabel("CDF")
        ax.set_title("CDF of latest RTT per flow")
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        return fig

    def _fig_box_latest_all(self):
        flows = getattr(self.ro, "flows", []) or []
        data, labels = [], []
        for f in flows:
            tl = getattr(f, "timeline", []) or []
            xs = [r["latest_rtt"] / 1000.0 for r in tl if r.get("latest_rtt") is not None]
            if xs:
                data.append(xs)
                labels.append((getattr(f, "flow_id", "flow") or "flow")[:8])
        if not data:
            return None
        fig, ax = plt.subplots()
        ax.boxplot(data, labels=labels, showfliers=self.showfliers)
        ax.set_ylabel("latest RTT (ms)")
        ax.set_title("Latest RTT distribution per flow (box plot)")
        fig.tight_layout()
        return fig

    # ---------------- PDF writers ----------------
    def _save_per_flow_pdfs(self):
        flows = getattr(self.ro, "flows", []) or []
        q = getattr(self.ro, "queue_series", None)
        for f in flows:
            fid = (getattr(f, "flow_id", "flow") or "flow")[:8]
            pdf_path = self.out_dir / f"{fid}_single.pdf"
            with PdfPages(pdf_path) as pdf:
                for builder in (#self._fig_owd,
                                lambda fl: self._fig_owd_queue(fl, q),
                                self._fig_cwnd_ssthresh_events,
                                self._fig_rtt_latest_min,
                                self._fig_bytes_cum):
                    fig = builder(f)
                    if fig is not None:
                        pdf.savefig(fig)
                        plt.close(fig)

    def _save_multiflow_pdf(self):
        multi_pdf = self.out_dir / "multiflow.pdf"
        with PdfPages(multi_pdf) as pdf:
            for fig in (self._fig_owd_all(),
                        self._fig_ecdf_latest_all(),
                        self._fig_box_latest_all()):
                if fig is not None:
                    pdf.savefig(fig)
                    plt.close(fig)

    # ---------------- CSV report ----------------
    def _per_flow_metrics(self, flow):
        fid = (getattr(flow, "flow_id", "flow") or "flow")[:8]
        duration_s = self._duration_seconds(flow) or 0.0
        ##tl = self._compute_bytes_acked_inline(getattr(flow, "timeline", []) or [])
        tl = getattr(flow, "timeline", []) or []
        bytes_sent  = sum(r.get("bytes_sent", 0)  or 0 for r in tl)
        bytes_acked = sum(r.get("bytes_acked", 0) or 0 for r in tl)
        bytes_lost  = sum(r.get("bytes_lost", 0)  or 0 for r in tl)
       

        throughput_Mbps = (bytes_sent  * 8.0) / duration_s / 1e6 if duration_s > 0 else 0.0
        goodput_Mbps    = (bytes_acked * 8.0) / duration_s / 1e6 if duration_s > 0 else 0.0
        loss_rate_pct   = 100.0 * (bytes_lost / float(bytes_sent)) if bytes_sent > 0 else 0.0

        _, owd_ms = self._get_run_owd_arrays_ms()
        owd_avg_ms, owd_std_ms = self._mean_std(owd_ms)

        latest_ms = [r["latest_rtt"] / 1000.0 for r in tl if r.get("latest_rtt") is not None]
        min_ms    = [r["min_rtt"]    / 1000.0 for r in tl if r.get("min_rtt")    is not None]
        rtt_latest_avg_ms, rtt_latest_std_ms = self._mean_std(latest_ms)
        rtt_min_ms = min(min_ms) if min_ms else ""

        return {
            "flow_id": fid,
            "duration_s": duration_s,
            "throughput_Mbps": throughput_Mbps,
            "goodput_Mbps": goodput_Mbps,
            "loss_rate_pct": loss_rate_pct,
            "owd_avg_ms": owd_avg_ms,
            "owd_std_ms": owd_std_ms,
            "rtt_latest_avg_ms": rtt_latest_avg_ms,
            "rtt_latest_std_ms": rtt_latest_std_ms,
            "rtt_min_ms": rtt_min_ms,
        }

    def _multi_flow_metrics(self, per_flow_rows):
        flows = getattr(self.ro, "flows", []) or []
        q = getattr(self.ro, "queue_series", None)

        q_avg_bytes = ""
        q_std_bytes = ""
        if q is not None:
            qbytes = getattr(q, "backlog_bytes", []) or []
            if qbytes:
                q_avg_bytes, q_std_bytes = self._mean_std(qbytes)

        all_latest_ms, all_min_ms = [], []

        _, all_owd_ms = self._get_run_owd_arrays_ms()

        for f in flows:
            tl = self._compute_bytes_acked_inline(getattr(f, "timeline", []) or [])
            all_latest_ms.extend([r["latest_rtt"] / 1000.0 for r in tl if r.get("latest_rtt") is not None])
            all_min_ms.extend([r["min_rtt"] / 1000.0 for r in tl if r.get("min_rtt") is not None])


        owd_avg_ms, owd_std_ms = self._mean_std(all_owd_ms)
        rtt_latest_avg_ms, rtt_latest_std_ms = self._mean_std(all_latest_ms)
        rtt_min_ms = min(all_min_ms) if all_min_ms else ""

        total_throughput = sum(r["throughput_Mbps"] for r in per_flow_rows if isinstance(r["throughput_Mbps"], (int, float)))
        total_goodput    = sum(r["goodput_Mbps"]    for r in per_flow_rows if isinstance(r["goodput_Mbps"], (int, float)))

        sum_sent = sum(sum(rr.get("bytes_sent", 0)  or 0 for rr in getattr(f, "timeline", []) or []) for f in flows)
        sum_lost = sum(sum(rr.get("bytes_lost", 0)  or 0 for rr in getattr(f, "timeline", []) or []) for f in flows)
        overall_loss_pct = 100.0 * (sum_lost / float(sum_sent)) if sum_sent > 0 else 0.0

        return {
            "flows_count": len(per_flow_rows),
            "queue_backlog_avg_bytes": q_avg_bytes,
            "queue_backlog_std_bytes": q_std_bytes,
            "owd_avg_ms": owd_avg_ms,
            "owd_std_ms": owd_std_ms,
            "rtt_latest_avg_ms": rtt_latest_avg_ms,
            "rtt_latest_std_ms": rtt_latest_std_ms,
            "rtt_min_ms": rtt_min_ms,
            "total_throughput_Mbps": total_throughput,
            "total_goodput_Mbps": total_goodput,
            "overall_loss_rate_pct": overall_loss_pct,
        }
    
    def _write_report_txt(self):
        """Write the same human-readable report that normally goes to stdout into report.txt."""
        txt_path = self.out_dir / "report.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            self.print_report(stream=f)
        return txt_path


    def _write_report_csv(self):
        flows = getattr(self.ro, "flows", []) or []
        per_flow_rows = [self._per_flow_metrics(f) for f in flows]
        multi = self._multi_flow_metrics(per_flow_rows)

        csv_path = self.out_dir / "report.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            # Per-flow table
            w.writerow(["flow_id","duration_s","throughput_Mbps","goodput_Mbps","loss_rate_pct",
                        "owd_avg_ms","owd_std_ms","rtt_latest_avg_ms","rtt_latest_std_ms","rtt_min_ms"])
            for r in per_flow_rows:
                w.writerow([
                    r["flow_id"],
                    f"{r['duration_s']:.6f}" if r["duration_s"] != "" else "",
                    f"{r['throughput_Mbps']:.6f}" if r["throughput_Mbps"] != "" else "",
                    f"{r['goodput_Mbps']:.6f}"    if r["goodput_Mbps"]    != "" else "",
                    f"{r['loss_rate_pct']:.3f}"   if r["loss_rate_pct"]   != "" else "",
                    f"{r['owd_avg_ms']:.6f}"      if r["owd_avg_ms"]      != "" else "",
                    f"{r['owd_std_ms']:.6f}"      if r["owd_std_ms"]      != "" else "",
                    f"{r['rtt_latest_avg_ms']:.6f}" if r["rtt_latest_avg_ms"] != "" else "",
                    f"{r['rtt_latest_std_ms']:.6f}" if r["rtt_latest_std_ms"] != "" else "",
                    f"{r['rtt_min_ms']:.6f}"      if r["rtt_min_ms"]      != "" else "",
                ])

            # Blank line
            w.writerow([])

            # Multi-flow table (single row)
            w.writerow(["flows_count","queue_backlog_avg_bytes","queue_backlog_std_bytes",
                        "owd_avg_ms","owd_std_ms","rtt_latest_avg_ms","rtt_latest_std_ms",
                        "rtt_min_ms","total_throughput_Mbps","total_goodput_Mbps","overall_loss_rate_pct"])
            w.writerow([
                multi["flows_count"],
                f"{multi['queue_backlog_avg_bytes']:.6f}" if multi["queue_backlog_avg_bytes"] != "" else "",
                f"{multi['queue_backlog_std_bytes']:.6f}" if multi["queue_backlog_std_bytes"] != "" else "",
                f"{multi['owd_avg_ms']:.6f}" if multi["owd_avg_ms"] != "" else "",
                f"{multi['owd_std_ms']:.6f}" if multi["owd_std_ms"] != "" else "",
                f"{multi['rtt_latest_avg_ms']:.6f}" if multi["rtt_latest_avg_ms"] != "" else "",
                f"{multi['rtt_latest_std_ms']:.6f}" if multi["rtt_latest_std_ms"] != "" else "",
                f"{multi['rtt_min_ms']:.6f}" if multi["rtt_min_ms"] != "" else "",
                f"{multi['total_throughput_Mbps']:.6f}",
                f"{multi['total_goodput_Mbps']:.6f}",
                f"{multi['overall_loss_rate_pct']:.3f}",
            ])
        return csv_path
