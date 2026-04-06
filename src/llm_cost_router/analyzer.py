"""Orchestration layer — ties parsing, classification, costing, and migration together."""

import structlog

from .classifier import classify_query
from .migration_planner import generate_migration_plan
from .models import AnalysisReport
from .parsers import detect_and_parse

log = structlog.get_logger()


def analyze(
    filepath: str,
    fmt: str | None = None,
    target_simple: str | None = None,
    target_medium: str | None = None,
) -> AnalysisReport:
    records = detect_and_parse(filepath, fmt)
    log.info("parsed usage logs", count=len(records))

    for rec in records:
        result = classify_query(rec)
        rec.complexity = result.complexity

    report = generate_migration_plan(
        records,
        target_simple=target_simple,
        target_medium=target_medium,
    )
    log.info("analysis complete", savings_pct=report.savings_pct)
    return report
