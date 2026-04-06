"""Generate Markdown and JSON reports from analysis results."""

import json
from collections import defaultdict

from .models import AnalysisReport


def generate_markdown_report(report: AnalysisReport) -> str:
    lines = [
        "# LLM Usage Analysis Report",
        "",
        "## Summary",
        f"- **Total queries analyzed:** {report.total_queries:,}",
    ]

    if report.date_range:
        lines.append(f"- **Date range:** {report.date_range[0]} to {report.date_range[1]}")

    lines.append(f"- **Current monthly cost:** ${report.current_total_cost:,.2f}")
    lines.append("")

    # Query distribution table
    lines.append("## Query Distribution")
    lines.append("| Complexity | Count | % of Total | Current Cost |")
    lines.append("|------------|------:|------------|-------------:|")

    for complexity in ["simple", "medium", "complex"]:
        count = report.distribution.get(complexity, 0)
        pct = (count / report.total_queries * 100) if report.total_queries else 0
        tier_cost = sum(
            t.current_cost for t in report.migration_plan if t.complexity == complexity
        )
        lines.append(
            f"| {complexity.capitalize():<10} | {count:>5,} | {pct:>9.1f}% | ${tier_cost:>10,.2f} |"
        )

    lines.append("")

    # Migration recommendations grouped by model
    lines.append("## Migration Recommendation")
    lines.append("| Current Model | Queries | Recommended | Monthly Savings |")
    lines.append("|---------------|--------:|-------------|----------------:|")

    by_model: dict[str, list] = defaultdict(list)
    for t in report.migration_plan:
        by_model[t.current_model].append(t)

    for model, targets in sorted(by_model.items()):
        total_q = sum(t.query_count for t in targets)
        recs = []
        for t in sorted(targets, key=lambda x: x.complexity):
            if t.recommended_model == t.current_model:
                recs.append(f"keep ({t.complexity})")
            else:
                recs.append(f"{t.recommended_model} ({t.complexity})")
        model_savings = sum(t.savings for t in targets)
        lines.append(
            f"| {model:<13} | {total_q:>7,} | {', '.join(recs)} | ${model_savings:>14,.2f} |"
        )

    lines.append("")

    # Projected savings
    lines.append("## Projected Savings")
    lines.append(f"- **Current monthly cost:** ${report.current_total_cost:,.2f}")
    lines.append(f"- **Projected monthly cost:** ${report.projected_total_cost:,.2f}")
    lines.append(
        f"- **Monthly savings:** ${report.total_savings:,.2f} ({report.savings_pct:.1f}%)"
    )
    lines.append(f"- **Annual savings:** ${report.total_savings * 12:,.2f}")
    lines.append("")

    return "\n".join(lines)


def generate_json_report(report: AnalysisReport) -> str:
    data = report.model_dump()
    return json.dumps(data, indent=2, default=str)
