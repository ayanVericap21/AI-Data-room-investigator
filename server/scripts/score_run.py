"""Score a run's findings against an answer key.

    scripts/score_run.py --key ../test_results/answer_keys/dive_room.json [--findings ../test_results/TEST-RUN-....txt]

Without --findings the newest TEST-RUN-*.txt in ../test_results is used (its findings are read from the FINDINGS block). Prints the report; nothing is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

from app.evaluation.scoring import by_family, load_findings, load_key, render, score  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True)
    ap.add_argument("--findings", default=None)
    args = ap.parse_args()
    results = SERVER_DIR.parent / "test_results"
    findings_path = Path(args.findings) if args.findings else max(results.glob("TEST-RUN-*.txt"), key=lambda p: p.stat().st_mtime, default=None)
    if findings_path is None or not findings_path.exists():
        print("no findings file found; run the investigator first", file=sys.stderr)
        return 2
    key = load_key(Path(args.key))
    s = score(key, load_findings(findings_path))
    text = render(s, str(findings_path))
    fam = by_family(s)
    if fam:
        text += "\nBy check family (issues expected to be found by that family, and how many it matched):\n" + "\n".join(f"  {k}: {v['found_by_that_family']} of {v['expected']}" for k, v in fam.items()) + "\n"
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
