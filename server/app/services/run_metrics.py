"""Where a run's metrics file lives, and how to find the one that belongs to an investigation."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import SERVER_DIR
from app.schemas.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

RESULTS_DIR = SERVER_DIR.parent / "test_results"
SUFFIX = ".metrics.json"


def metrics_path_for(results_file: Path) -> Path:
    """TEST-RUN-….txt -> TEST-RUN-….metrics.json (next to it)."""
    return results_file.with_name(results_file.stem + SUFFIX)


def write_metrics(results_file: Path, metrics: RunMetrics) -> Path:
    path = metrics_path_for(results_file)
    path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    return path


def extract_block(text: str, name: str) -> str | None:
    """The text between a ``<<<NAME`` line and a ``NAME>>>`` line: a run that keeps everything in its one results file embeds its JSON this way."""
    start, end = f"<<<{name}\n", f"\n{name}>>>"
    a = text.find(start)
    if a < 0:
        return None
    b = text.find(end, a + len(start))
    return text[a + len(start):b] if b >= 0 else None


def find_metrics(investigation_id: str, directory: Path | None = None) -> RunMetrics | None:
    """The newest metrics written for this investigation (a ``.metrics.json`` file, or the RUN_METRICS block inside a results text file), or None."""
    directory = directory or RESULTS_DIR
    if not directory.is_dir():
        return None
    files = sorted([*directory.glob(f"*{SUFFIX}"), *directory.glob("TEST-RUN-*.txt")], key=lambda p: p.name, reverse=True)
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8")
            if path.suffix == ".txt":
                raw = extract_block(raw, "RUN_METRICS")
                if raw is None:
                    continue
            data = json.loads(raw)
        except (OSError, ValueError):
            logger.warning("Unreadable metrics file", extra={"file": str(path)})
            continue
        if data.get("investigation_id") == investigation_id:
            return RunMetrics.model_validate(data)
    return None
