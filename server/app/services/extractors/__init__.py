"""Format-specific extractors. ``lap`` is the shared helper that records per-part timings."""
import time

from app.schemas.extraction import ExtractionReport


def lap(report: ExtractionReport | None, key: str, started: float) -> None:
    """Add the milliseconds since ``started`` (a perf_counter value) to ``report.timings_ms[key]``, if a report is wanted."""
    if report is not None:
        report.timings_ms[key] = report.timings_ms.get(key, 0) + round((time.perf_counter() - started) * 1000)
