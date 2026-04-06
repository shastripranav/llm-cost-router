"""Built-in LLM provider pricing tables (per 1M tokens, early 2026).

Users can override with custom pricing via --update flag or by passing
a JSON file with the same structure.
"""

import json
from pathlib import Path

import structlog

log = structlog.get_logger()

# per-1M pricing avoids floating point issues with per-token rates
PROVIDER_PRICING: dict[str, dict] = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "provider": "openai"},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "provider": "openai"},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "provider": "openai"},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50, "provider": "openai"},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00, "provider": "anthropic"},
    "claude-3-haiku": {"input": 0.25, "output": 1.25, "provider": "anthropic"},
    "claude-3.5-haiku": {"input": 0.80, "output": 4.00, "provider": "anthropic"},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00, "provider": "google"},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30, "provider": "google"},
    "mistral-large": {"input": 2.00, "output": 6.00, "provider": "mistral"},
    "mistral-small": {"input": 0.20, "output": 0.60, "provider": "mistral"},
    "llama-3.1-70b": {"input": 0.00, "output": 0.00, "provider": "meta", "note": "self-hosted"},
    "llama-3.1-8b": {"input": 0.00, "output": 0.00, "provider": "meta", "note": "self-hosted"},
}


def get_pricing(custom_path: str | None = None) -> dict[str, dict]:
    pricing = {k: dict(v) for k, v in PROVIDER_PRICING.items()}

    if custom_path:
        path = Path(custom_path)
        if not path.exists():
            raise FileNotFoundError(f"Custom pricing file not found: {path}")

        with open(path) as f:
            overrides = json.load(f)

        for model, rates in overrides.items():
            if model in pricing:
                pricing[model].update(rates)
                log.debug("pricing override applied", model=model)
            else:
                pricing[model] = rates
                log.debug("custom model added", model=model)

    return pricing


def load_custom_pricing(filepath: str) -> dict:
    """Load and validate a custom pricing JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    for model, rates in data.items():
        if "input" not in rates or "output" not in rates:
            raise ValueError(f"Model '{model}' missing required 'input'/'output' pricing fields")

    return data
