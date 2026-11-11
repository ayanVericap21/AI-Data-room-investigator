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


def find_metrics(investigation_id: str, directory: Path | None = None) -> RunMetrics | None:
    """The newest metrics file written for this investigation, or None (older runs have none)."""
    directory = directory or RESULTS_DIR
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob(f"*{SUFFIX}"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Unreadable metrics file", extra={"file": str(path)})
            continue
        if data.get("investigation_id") == investigation_id:
            return RunMetrics.model_validate(data)
    return None
