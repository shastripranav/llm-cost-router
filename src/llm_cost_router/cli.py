"""CLI interface for LLM Cost Router.

Commands: analyze, classify, migrate, savings, pricing
"""

import json

import click
from rich.console import Console
from rich.table import Table

from .analyzer import analyze
from .classifier import classify_query
from .config import setup_logging
from .parsers import detect_and_parse
from .pricing import PROVIDER_PRICING, get_pricing, load_custom_pricing
from .report_generator import generate_json_report, generate_markdown_report

console = Console(stderr=True)
out = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose):
    """LLM Cost Router — analyze usage, classify queries, plan migrations."""
    setup_logging(verbose=verbose)


@cli.command("analyze")
@click.argument("logfile", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["openai", "csv", "anthropic"]))
@click.option("--output", "-o", type=click.Choice(["markdown", "json"]), default="markdown")
def analyze_cmd(logfile, fmt, output):
    """Analyze usage patterns and generate a cost report."""
    report = analyze(logfile, fmt)
    if output == "json":
        out.print(generate_json_report(report))
    else:
        out.print(generate_markdown_report(report))


@cli.command("classify")
@click.argument("logfile", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["openai", "csv", "anthropic"]))
@click.option("--output", "-o", type=click.Path(), help="Write classified results to JSON file")
def classify_cmd(logfile, fmt, output):
    """Classify queries by complexity."""
    records = detect_and_parse(logfile, fmt)

    results = []
    for rec in records:
        result = classify_query(rec)
        rec.complexity = result.complexity
        results.append(
            {
                "model": rec.model,
                "prompt_tokens": rec.prompt_tokens,
                "completion_tokens": rec.completion_tokens,
                "complexity": result.complexity,
                "confidence": result.confidence,
                "signals": result.signals,
            }
        )

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Classification written to {output}[/green]")
        return

    table = Table(title="Query Classification")
    table.add_column("Model", style="cyan")
    table.add_column("Prompt Tok", justify="right")
    table.add_column("Complexity")
    table.add_column("Confidence", justify="right")
    table.add_column("Signals")

    # TODO: add support for streaming API logs (chunked responses)
    for r in results[:30]:
        color = {"simple": "green", "medium": "yellow", "complex": "red"}.get(
            r["complexity"], "white"
        )
        table.add_row(
            r["model"],
            str(r["prompt_tokens"]),
            f"[{color}]{r['complexity']}[/{color}]",
            f"{r['confidence']:.0%}",
            ", ".join(r["signals"][:3]),
        )

    if len(results) > 30:
        table.caption = f"Showing 30 of {len(results)} queries"

    out.print(table)


@cli.command("migrate")
@click.argument("logfile", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["openai", "csv", "anthropic"]))
@click.option("--target-simple", default="gpt-4o-mini", help="Target model for simple queries")
@click.option("--target-medium", default="claude-3.5-haiku", help="Target model for medium queries")
@click.option("--output", "-o", type=click.Choice(["markdown", "json"]), default="markdown")
def migrate_cmd(logfile, fmt, target_simple, target_medium, output):
    """Generate a migration plan with custom target models."""
    report = analyze(logfile, fmt, target_simple=target_simple, target_medium=target_medium)
    if output == "json":
        out.print(generate_json_report(report))
    else:
        out.print(generate_markdown_report(report))


@cli.command("savings")
@click.argument("logfile", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["openai", "csv", "anthropic"]))
def savings_cmd(logfile, fmt):
    """Quick savings estimate for a log file."""
    report = analyze(logfile, fmt)

    table = Table(title="Savings Estimate", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Queries", f"{report.total_queries:,}")
    table.add_row("Current Monthly Cost", f"${report.current_total_cost:,.2f}")
    table.add_row("Projected Monthly Cost", f"${report.projected_total_cost:,.2f}")
    table.add_row(
        "Monthly Savings",
        f"[green]${report.total_savings:,.2f} ({report.savings_pct:.1f}%)[/green]",
    )
    table.add_row(
        "Annual Savings",
        f"[green bold]${report.total_savings * 12:,.2f}[/green bold]",
    )

    out.print(table)


@cli.command("pricing")
@click.option("--update", type=click.Path(exists=True), help="Merge custom pricing from JSON file")
def pricing_cmd(update):
    """Show built-in pricing table or merge custom overrides."""
    if update:
        custom = load_custom_pricing(update)
        merged = get_pricing(update)
        console.print(f"[green]Merged {len(custom)} model(s) into pricing table[/green]")
        _display_pricing(merged)
    else:
        _display_pricing(PROVIDER_PRICING)


def _display_pricing(pricing: dict):
    table = Table(title="LLM Pricing (per 1M tokens)")
    table.add_column("Model", style="cyan")
    table.add_column("Provider")
    table.add_column("Input $", justify="right")
    table.add_column("Output $", justify="right")
    table.add_column("Notes")

    for model, rates in sorted(pricing.items()):
        table.add_row(
            model,
            rates.get("provider", "—"),
            f"${rates['input']:.3f}",
            f"${rates['output']:.3f}",
            rates.get("note", ""),
        )

    out.print(table)
