"""Parse Anthropic API log format.

Handles both the Messages API response shape and simplified log entries.
Anthropic uses input_tokens/output_tokens instead of prompt_tokens/completion_tokens.
"""

import json
from datetime import datetime, timezone

from ..models import QueryRecord


def parse_anthropic_logs(filepath: str) -> list[QueryRecord]:
    with open(filepath) as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    records = []
    for entry in data:
        usage = entry.get("usage", {})
        records.append(
            QueryRecord(
                timestamp=_parse_ts(entry),
                model=entry["model"],
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                prompt_text=_extract_prompt(entry),
                response_text=_extract_response(entry),
            )
        )

    return records


def _parse_ts(entry: dict) -> datetime | None:
    raw = entry.get("created_at") or entry.get("timestamp")
    if raw is None:
        return None
    if isinstance(raw, str):
        return datetime.fromisoformat(raw)
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    return None


def _extract_prompt(entry: dict) -> str | None:
    messages = entry.get("messages")
    if not messages:
        return None
    parts = []
    for m in messages:
        content = m.get("content", "")
        # FIXME: this flattens content blocks but drops image blocks entirely
        if isinstance(content, list):
            content = " ".join(
                blk.get("text", "")
                for blk in content
                if isinstance(blk, dict) and blk.get("type") == "text"
            )
        if content:
            parts.append(content)
    return "\n".join(parts) if parts else None


def _extract_response(entry: dict) -> str | None:
    content = entry.get("content")
    if content is None:
        return None
    if isinstance(content, list):
        texts = [
            blk.get("text", "")
            for blk in content
            if isinstance(blk, dict) and blk.get("type") == "text"
        ]
        return " ".join(texts) if texts else None
    if isinstance(content, str):
        return content
    return None
