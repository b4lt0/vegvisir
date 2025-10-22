import argparse
from pathlib import Path
from datetime import datetime
from netan import RunProcessor
from netan.report_generator import ReportGenerator


parser = argparse.ArgumentParser(description="Compute CC metrics using process_run (qlog/owd/queue).")
parser.add_argument("--qlog", action="append", default=[], help="qlog path (repeat once per flow)")
parser.add_argument("--owd", required=True, help="OWD file path")
parser.add_argument("--queue-run-dir", required=True, help="Directory containing queue logs")
parser.add_argument("--device", default="eth0")
parser.add_argument("--qdisc", default="netem")
parser.add_argument("-v", "--verbose", action="store_true", help="Print human-readable metrics to stdout")
args = parser.parse_args()

qlogs = args.qlog or []
owd  = args.owd or ""

rp = RunProcessor(queue_run_dir=args.queue_run_dir, device=args.device, qdisc=args.qdisc)
for q in args.qlog:
    rp.add_qlog(q)
rp.set_owd_file(owd)

run_output = rp.run()

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path(f"{stamp}_stats")

# generate reports (PDFs + CSV). Returns the CSV path.
rg = ReportGenerator(run_output, out_dir, show_p50=True, show_p90=True, showfliers=False)
csv_path=rg.generate()

if args.verbose:
    rg.print_report()

print("Reports written to:", out_dir)