import json
from pathlib import Path

import structlog

from ..models import QueryRecord
from .anthropic_parser import parse_anthropic_logs
from .csv_parser import parse_csv_logs
from .openai_parser import parse_openai_logs

log = structlog.get_logger()

FORMAT_PARSERS = {
    "openai": parse_openai_logs,
    "csv": parse_csv_logs,
    "anthropic": parse_anthropic_logs,
}


def detect_format(filepath: str) -> str:
    """Auto-detect log format from file extension and content structure."""
    path = Path(filepath)

    if path.suffix == ".csv":
        return "csv"

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)

        sample = data[0] if isinstance(data, list) else data
        usage = sample.get("usage", {})

        # Anthropic uses input_tokens/output_tokens
        if "input_tokens" in usage:
            return "anthropic"

        return "openai"

    raise ValueError(
        f"Cannot auto-detect format for '{path.suffix}' files. Use --format to specify."
    )


def detect_and_parse(filepath: str, fmt: str | None = None) -> list[QueryRecord]:
    if fmt is None:
        fmt = detect_format(filepath)

    parser = FORMAT_PARSERS.get(fmt)
    if parser is None:
        raise ValueError(f"Unknown format: {fmt}. Supported: {', '.join(FORMAT_PARSERS)}")

    records = parser(filepath)
    log.debug("parsed log file", format=fmt, records=len(records))
    return records
