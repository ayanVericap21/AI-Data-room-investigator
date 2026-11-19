"""Mini corpus test: the whole pipeline on a short sheet of text, each paragraph a document of its own.

    scripts/mini_run.py corpus.txt            (or  -  to read the sheet from the standard input)
    scripts/mini_run.py corpus.txt -- --claim-filter     (anything after -- goes to ledger_run.py)

Paragraphs are separated by blank lines (any number, 1 to as many as given). Each becomes a small Word file (P01.docx, P02.docx, ...) in a scratch data
room, and the real pipeline runs on them in a scratch database that is created for the run and dropped at its end: parsing and chunking, LlamaExtract
(a few credits per paragraph), the entity gate, grouping (with rules), the judge, and the missing-item and identifier checks. Nothing in the real
data room or the real database is touched.

The report (printed and written to ../test_results/MINI-<stamp>.txt) is: the paragraph each document holds, then the pipeline's own report, which ends with every claim
extracted from every document (the pipeline's own file, TEST-RUN-<stamp>-mini.txt, has the same report without the paragraphs). Nothing here decides what is right; the report is for whoever evaluates it.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SERVER_DIR.parent
sys.path.insert(0, str(SERVER_DIR))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

from app.core.config import get_settings  # noqa: E402

RESULTS_LINE = re.compile(r"\[run\] results file=(.+)")


def paragraphs(sheet: str) -> list[str]:
    """The sheet's paragraphs: blocks of text separated by blank lines, whitespace inside a block folded to single spaces."""
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", sheet.strip()) if block.strip()]


def write_documents(paras: list[str], folder: Path) -> list[str]:
    import docx  # python-docx, already a dependency of the document parser

    names = []
    for n, para in enumerate(paras, start=1):
        name = f"P{n:02d}.docx"
        document = docx.Document()
        document.add_paragraph(para)
        document.save(folder / name)
        names.append(name)
    return names


def admin_engine(url: str):
    return create_engine(make_url(url).set(database="postgres"), isolation_level="AUTOCOMMIT")


def create_scratch_database(url: str, name: str) -> str:
    admin = admin_engine(url)
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    scratch = make_url(url).set(database=name).render_as_string(hide_password=False)
    done = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=SERVER_DIR, env={**os.environ, "DATABASE_URL": scratch},
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError("could not create the scratch tables:\n" + done.stderr[-800:])
    return scratch


def drop_scratch_database(url: str, name: str) -> None:
    admin = admin_engine(url)
    with admin.connect() as conn:
        conn.execute(text("select pg_terminate_backend(pid) from pg_stat_activity where datname = :d and pid <> pg_backend_pid()"), {"d": name})
        conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
    admin.dispose()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="a text file of paragraphs separated by blank lines, or - for the standard input")
    ap.add_argument("--keep", action="store_true", help="keep the scratch database and files (their names are printed)")
    ap.add_argument("passthrough", nargs=argparse.REMAINDER, help="after --, options for ledger_run.py")
    args = ap.parse_args()
    sheet = sys.stdin.read() if args.corpus == "-" else Path(args.corpus).read_text(encoding="utf-8")
    paras = paragraphs(sheet)
    if not paras:
        print("the corpus has no paragraphs", file=sys.stderr)
        return 2
    extra = [a for a in args.passthrough if a != "--"]

    settings = get_settings()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    db_name = f"mini_{stamp.replace('-', '_')}"
    work = Path(tempfile.mkdtemp(prefix="mini_corpus_"))
    (work / "data_room").mkdir()
    files = write_documents(paras, work / "data_room")
    print(f"[mini] {len(paras)} paragraphs -> {len(files)} documents in {work / 'data_room'}", flush=True)
    scratch_url = None
    started = time.perf_counter()
    try:
        scratch_url = create_scratch_database(settings.database_url, db_name)
        print(f"[mini] scratch database {db_name} ready; running the pipeline", flush=True)
        command = [sys.executable, "-u", str(SERVER_DIR / "scripts" / "ledger_run.py"), "--config", "mini", "--judge-workers", "2", "--dump-claims", "--save-consistent"]
        for name in files:
            command += ["--only", name]
        env = {**os.environ, "DATABASE_URL": scratch_url, "DATA_ROOM_PATH": str(work / "data_room")}
        process = subprocess.Popen(command + extra, cwd=SERVER_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        results_file, log_tail = None, []
        for line in process.stdout:
            line = line.rstrip("\n")
            if "Claim rejected" in line or "Transient LLM error" in line:
                continue
            print(line, flush=True)
            log_tail = (log_tail + [line])[-30:]
            if (m := RESULTS_LINE.search(line)):
                results_file = Path(m.group(1).strip())
        process.wait()
        report = results_file.read_text(encoding="utf-8") if results_file and results_file.exists() else "THE PIPELINE WROTE NO REPORT. Last lines:\n" + "\n".join(log_tail)
        header = [f"MINI CORPUS TEST  {stamp}", "=" * 78, f"{len(paras)} paragraphs; each is a document of its own (P01 ... P{len(paras):02d}); wall time {time.perf_counter() - started:.0f}s",
                  f"pipeline exit code {process.returncode}; options passed to ledger_run.py: {extra or 'none'}", "", "THE DOCUMENTS (paragraph = document)", "=" * 78]
        for name, para in zip(files, paras):
            header += [f"{name}: {para}", ""]
        body = "\n".join(header) + "\n\nTHE PIPELINE'S REPORT\n" + "=" * 78 + "\n" + report
        out = REPO_DIR / "test_results" / f"MINI-{stamp}.txt"
        out.write_text(body, encoding="utf-8")
        print(f"[mini] report: {out}", flush=True)
        return process.returncode
    finally:
        if args.keep:
            print(f"[mini] kept: database {db_name}, files {work}", flush=True)
        else:
            if scratch_url is not None:
                drop_scratch_database(settings.database_url, db_name)
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
