"""Parse OpenAI API log format.

Expected structure per entry:
  {
    "model": "gpt-4o",
    "created": 1706745600,
    "usage": {"prompt_tokens": 45, "completion_tokens": 120},
    "messages": [...],
    "choices": [{"message": {"role": "assistant", "content": "..."}}]
  }
"""

import json
from datetime import datetime, timezone

import structlog

from ..models import QueryRecord

log = structlog.get_logger()


def parse_openai_logs(filepath: str) -> list[QueryRecord]:
    with open(filepath) as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    records = []
    for i, entry in enumerate(data):
        try:
            usage = entry.get("usage", {})
            record = QueryRecord(
                timestamp=_parse_created(entry.get("created")),
                model=entry["model"],
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                prompt_text=_extract_prompt(entry),
                response_text=_extract_response(entry),
            )
            records.append(record)
        except (KeyError, TypeError) as exc:
            log.warning("skipping malformed entry", index=i, error=str(exc))

    return records


def _parse_created(ts) -> datetime | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _extract_prompt(entry: dict) -> str | None:
    messages = entry.get("messages")
    if not messages:
        return None
    parts = [m.get("content", "") for m in messages if m.get("role") != "assistant"]
    return "\n".join(parts) if parts else None


def _extract_response(entry: dict) -> str | None:
    choices = entry.get("choices")
    if not choices:
        return None
    msg = choices[0].get("message", {})
    return msg.get("content")
