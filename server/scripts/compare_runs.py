"""Compare ledger runs against the answer key, case by case, and write everything to ONE file in test_results.

    python scripts/compare_runs.py REPORT1.txt REPORT2.txt REPORT3.txt [--key KEY.csv] [--set HTC01] [--out test_results/NAME.txt]

Each report is read for its validity banner, its findings (title, class, severity, documents, quotes) and its counts. A finding is matched to a case of
the key when its text (description, evidence quotes) holds enough of the case's own figures, dates and quoted words AND its evidence comes from the
documents the case cites. This is an automatic, approximate match: the file lists the finding each match used, so a human can confirm it. The result
is a section for each run, then a table with every case: caught or not in each run, and how many times out of the runs. Cases that are consistent in
the key are traps: a finding that matches one is a false alarm. Reports are appended in the order given; nothing is sent to any model.
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_KEY = Path.home() / "Downloads" / "AI Data Room Evaluation Data Templates - Ground Truth.csv"
FINDING = re.compile(r"^--- FINDING (\d+) of \d+\s+\[(\w+)\]")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
MONTHS = "january february march april may june july august september october november december".split()


@dataclass
class Finding:
    id: str
    title: str = ""
    classification: str = ""
    severity: str = ""
    description: str = ""
    docs: set = field(default_factory=set)
    quotes: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.title} {self.description} {' '.join(self.quotes)}"


@dataclass
class Run:
    path: Path
    started: str = ""
    wall: str = ""
    valid: bool = True
    banner: str = ""
    summary: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)


def read_report(path: Path) -> Run:
    run = Run(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[:60]:
        if "INVALID FOR SCORING" in line:
            run.valid, run.banner = False, line.strip(" <>")
        if line.startswith("Run started"):
            run.started = line.split(":", 1)[1].split("finished")[0].strip()
            run.wall = line.rsplit("wall time:", 1)[-1].strip()
        for key in ("Claim groups", "Findings", "LLM calls", "Tokens"):
            if line.startswith(key + " "):
                run.summary[key] = line.split(":", 1)[1].strip()[:200]
    current = None
    for line in lines:
        m = FINDING.match(line)
        if m:
            current = Finding(m.group(2))
            run.findings.append(current)
            continue
        if line.startswith("====") and current is not None and not line.startswith("=== "):
            pass
        if current is None:
            continue
        stripped = line.strip()
        for field_name in ("title", "classification", "severity", "description"):
            if line.startswith(f"{field_name}") and " : " in line[:26]:
                setattr(current, field_name, line.split(" : ", 1)[1].strip())
        if stripped.startswith("document :"):
            current.docs.add(re.sub(r"_.*", "", stripped.split(":", 1)[1].strip()))
        elif stripped.startswith("quote    :"):
            current.quotes.append(stripped.split(":", 1)[1].strip().strip('"'))
    return run


def tokens_of(case: dict) -> tuple[set, set]:
    """(figures and dates, distinctive words from the quotes) the case is made of, and the documents it cites."""
    text = f"{case['Claims compared']} {case['Citations']}"
    figures = {re.sub(r"\D", "", n) for n in NUMBER.findall(text) if len(re.sub(r"\D", "", n)) >= 3}
    figures |= {f"{m}{d}" for m in MONTHS for d in re.findall(rf"{m}\s+(\d{{1,2}})", text.lower())}
    quoted = " ".join(re.findall(r'"([^"]+)"', case["Citations"]))
    words = {w for w in re.findall(r"[a-z]{5,}", quoted.lower()) if w not in {"which", "there", "would", "their", "other", "these", "shall"}}
    return figures, words


def matches(case: dict, finding: Finding) -> bool:
    figures, words = tokens_of(case)
    docs = set(re.findall(r"DOC-\d+", case["Citations"] + case["Documents"]))
    text = finding.text.lower()
    digits = re.sub(r"[^\d ]", "", text.replace(",", ""))
    have_figures = {f for f in figures if f in digits.replace(" ", "") or f in re.sub(r"\D", "", text)}
    have_words = {w for w in words if w in text}
    if len(docs) >= 2 and len(finding.docs & docs) < 2:
        return False
    if figures:  # a case made of figures needs two of them (one when it has only one) in the finding
        return len(have_figures) >= min(2, len(figures))
    return len(have_words) >= 4  # a case made only of words (a missing document) needs four of its own words


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--set", default="HTC01", help="the case id prefix of the data room")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cases = [r for r in csv.DictReader(args.key.open(encoding="utf-8-sig")) if r["Case ID"].startswith(args.set)]
    runs = [read_report(p) for p in args.reports]
    out = args.out or Path(__file__).resolve().parents[2] / "test_results" / f"RUN-COMPARISON-{datetime.now():%Y%m%d-%H%M%S}.txt"
    lines = [f"RUN COMPARISON against {args.key.name} ({len(cases)} cases of {args.set}), {datetime.now():%Y-%m-%d %H:%M:%S}",
             "Matches are automatic and approximate: each lists the finding it used, so a case can be confirmed by reading that finding.", ""]
    caught: dict[str, list] = {c["Case ID"]: [] for c in cases}
    for n, run in enumerate(runs, start=1):
        flagged = [c for c in cases if c["Classification"] != "Consistent"]
        traps = [c for c in cases if c["Classification"] == "Consistent"]
        hit, false_alarm = {}, {}
        for c in flagged:
            f = next((f for f in run.findings if matches(c, f)), None)
            if f:
                hit[c["Case ID"]] = f
        for c in traps:
            f = next((f for f in run.findings if matches(c, f) and f.classification in ("contradiction", "contextual_difference")), None)
            if f:
                false_alarm[c["Case ID"]] = f
        for c in cases:
            caught[c["Case ID"]].append(c["Case ID"] in hit if c["Classification"] != "Consistent" else c["Case ID"] not in false_alarm)
        lines += [f"=== RUN {n}: {run.path.name}", f"started {run.started}; wall time {run.wall}; valid for scoring: {'yes' if run.valid else 'NO'}" + ("" if run.valid else f" ({run.banner[:160]})"),
                  *[f"{k}: {v}" for k, v in run.summary.items()], f"findings read: {len(run.findings)}",
                  f"cases to flag caught: {len(hit)} of {len(flagged)}; traps (consistent cases) wrongly flagged: {len(false_alarm)} of {len(traps)}",
                  f"score as accuracy (caught + traps not flagged) / all cases: {(len(hit) + len(traps) - len(false_alarm)) / len(cases):.0%}"]
        for cid, f in sorted(hit.items()):
            case = next(c for c in cases if c["Case ID"] == cid)
            lines.append(f"   caught {cid} [{case['Classification']}] by [{f.id}] {f.classification}/{f.severity}: {f.title[:90]}")
        for cid, f in sorted(false_alarm.items()):
            lines.append(f"   FALSE ALARM {cid} (consistent in the key) by [{f.id}] {f.classification}/{f.severity}: {f.title[:90]}")
        lines.append("")
    lines += ["=== EVERY CASE ACROSS THE RUNS (caught / not flagged = yes; for a consistent case, yes means it was NOT flagged)",
              f"{'case':<11}{'key class':<24}{'severity':<10}" + "".join(f"run{n:<4}" for n in range(1, len(runs) + 1)) + "times  summary"]
    for c in cases:
        marks = caught[c["Case ID"]]
        lines.append(f"{c['Case ID']:<11}{c['Classification']:<24}{c['Severity']:<10}" + "".join(f"{'yes' if m else 'no':<7}" for m in marks) + f"{sum(marks)}/{len(marks)}    {c['Summary'][:70]}")
    stable = [c["Case ID"] for c in cases if c["Classification"] != "Consistent" and all(caught[c["Case ID"]])]
    sometimes = [c["Case ID"] for c in cases if c["Classification"] != "Consistent" and 0 < sum(caught[c["Case ID"]]) < len(runs)]
    never = [c["Case ID"] for c in cases if c["Classification"] != "Consistent" and not any(caught[c["Case ID"]])]
    lines += ["", f"caught in every run ({len(stable)}): {', '.join(stable) or 'none'}", f"caught in some runs only ({len(sometimes)}): {', '.join(sometimes) or 'none'}",
              f"never caught ({len(never)}): {', '.join(never) or 'none'}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4 + 6 * len(runs)]))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
