from collections import defaultdict

import structlog

from .cost_engine import aggregate_costs, calculate_query_cost
from .models import AnalysisReport, MigrationTarget, QueryRecord
from .pricing import get_pricing

log = structlog.get_logger()

DEFAULT_TARGETS: dict[str, list[str]] = {
    "simple": ["gpt-4o-mini", "claude-3-haiku", "llama-3.1-8b"],
    "medium": ["gpt-4o-mini", "claude-3.5-haiku", "mistral-small"],
    "complex": [],
}

BUDGET_MODELS = frozenset({
    "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-haiku",
    "gemini-1.5-flash", "mistral-small",
    "llama-3.1-8b", "llama-3.1-70b",
})


def recommend_target(
    current_model: str,
    complexity: str,
    overrides: dict[str, str] | None = None,
) -> str:
    if complexity == "complex":
        return current_model

    if current_model in BUDGET_MODELS:
        return current_model

    if overrides and complexity in overrides:
        return overrides[complexity]

    candidates = DEFAULT_TARGETS.get(complexity, [])
    return candidates[0] if candidates else current_model


def generate_migration_plan(
    records: list[QueryRecord],
    pricing: dict | None = None,
    target_simple: str | None = None,
    target_medium: str | None = None,
) -> AnalysisReport:
    """Build a full migration plan with cost projections.

    Groups records by (model, complexity), maps each group to a recommended
    target model, then calculates current vs projected costs.
    """
    if pricing is None:
        pricing = get_pricing()

    overrides: dict[str, str] = {}
    if target_simple:
        overrides["simple"] = target_simple
    if target_medium:
        overrides["medium"] = target_medium

    groups: dict[tuple[str, str], list[QueryRecord]] = defaultdict(list)
    for rec in records:
        key = (rec.model, rec.complexity or "medium")
        groups[key].append(rec)

    migration_targets = []
    current_total = 0.0
    projected_total = 0.0

    for (model, complexity), group_records in groups.items():
        target_model = recommend_target(model, complexity, overrides)

        cur_cost = sum(
            calculate_query_cost(model, r.prompt_tokens, r.completion_tokens, pricing)
            for r in group_records
        )
        proj_cost = sum(
            calculate_query_cost(target_model, r.prompt_tokens, r.completion_tokens, pricing)
            for r in group_records
        )

        current_total += cur_cost
        projected_total += proj_cost
        savings = cur_cost - proj_cost
        pct = (savings / cur_cost * 100) if cur_cost > 0 else 0.0

        migration_targets.append(MigrationTarget(
            current_model=model,
            complexity=complexity,
            recommended_model=target_model,
            query_count=len(group_records),
            current_cost=round(cur_cost, 4),
            projected_cost=round(proj_cost, 4),
            savings=round(savings, 4),
            savings_pct=round(pct, 1),
        ))

    total_savings = current_total - projected_total

    distribution: dict[str, int] = defaultdict(int)
    for rec in records:
        distribution[rec.complexity or "unknown"] += 1

    timestamps = [r.timestamp for r in records if r.timestamp]
    date_range = None
    if timestamps:
        date_range = (
            min(timestamps).strftime("%Y-%m-%d"),
            max(timestamps).strftime("%Y-%m-%d"),
        )

    return AnalysisReport(
        total_queries=len(records),
        date_range=date_range,
        current_total_cost=round(current_total, 2),
        projected_total_cost=round(projected_total, 2),
        total_savings=round(total_savings, 2),
        savings_pct=round((total_savings / current_total * 100) if current_total > 0 else 0.0, 1),
        distribution=dict(distribution),
        cost_by_model=aggregate_costs(records, pricing),
        migration_plan=sorted(migration_targets, key=lambda t: t.savings, reverse=True),
    )
