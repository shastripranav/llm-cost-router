from collections import defaultdict

import structlog

from .models import CostEstimate, QueryRecord
from .pricing import get_pricing

log = structlog.get_logger()


def calculate_query_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: dict | None = None,
) -> float:
    """Calculate cost for a single query based on model pricing and token counts."""
    if pricing is None:
        pricing = get_pricing()

    model_pricing = pricing.get(model)
    if model_pricing is None:
        log.warning("unknown model in pricing table, cost set to zero", model=model)
        return 0.0

    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    return input_cost + output_cost


def aggregate_costs(
    records: list[QueryRecord],
    pricing: dict | None = None,
) -> list[CostEstimate]:
    """Aggregate token usage and costs grouped by model.

    Args:
        records: List of parsed query records.
        pricing: Optional pricing dict override (uses built-in if None).

    Returns:
        Sorted list of CostEstimate objects, one per model.
    """
    if pricing is None:
        pricing = get_pricing()

    buckets: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "input_tok": 0, "output_tok": 0, "cost": 0.0}
    )

    for rec in records:
        b = buckets[rec.model]
        b["count"] += 1
        b["input_tok"] += rec.prompt_tokens
        b["output_tok"] += rec.completion_tokens
        b["cost"] += calculate_query_cost(
            rec.model, rec.prompt_tokens, rec.completion_tokens, pricing
        )

    estimates = []
    for model, d in sorted(buckets.items()):
        avg = d["cost"] / d["count"] if d["count"] else 0.0
        estimates.append(
            CostEstimate(
                model=model,
                total_queries=d["count"],
                total_input_tokens=d["input_tok"],
                total_output_tokens=d["output_tok"],
                total_cost=round(d["cost"], 6),
                avg_cost_per_query=round(avg, 6),
            )
        )

    return estimates
