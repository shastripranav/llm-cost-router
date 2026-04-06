# LLM Cost Router & Migration Toolkit

Analyze LLM API usage logs, classify query complexity, and generate migration plans with cost savings estimates.

This is the **analysis and planning layer** that runs before you set up routing — answering: "Which of my queries can safely move to a cheaper model, and how much will I save?"

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Analyze usage and get a full report
llm-router analyze logs.json

# Quick savings estimate
llm-router savings logs.json

# Classify queries by complexity
llm-router classify logs.json --output classified.json

# Generate migration plan with custom targets
llm-router migrate logs.json --target-simple gpt-4o-mini --target-medium claude-3.5-haiku

# View built-in pricing table
llm-router pricing
```

## Supported Log Formats

| Format | File Type | Auto-detected |
|--------|-----------|---------------|
| OpenAI | JSON with `usage.prompt_tokens` / `usage.completion_tokens` | Yes |
| Anthropic | JSON with `usage.input_tokens` / `usage.output_tokens` | Yes |
| CSV | `model, prompt_tokens, completion_tokens` + optional columns | Yes |

Format is auto-detected from file extension and content structure. Override with `--format openai|csv|anthropic`.

## How It Works

```
Usage Logs → Parse → Classify → Cost → Migrate → Report
               │         │         │        │         │
          OpenAI/CSV/  simple/   per-1M   model    Markdown
          Anthropic    medium/   token    mapping   or JSON
                       complex   pricing
```

**1. Parse** — Reads OpenAI, Anthropic, or CSV log files into normalized records.

**2. Classify** — Heuristic classifier assigns `simple` / `medium` / `complex` based on:
- Token counts (prompt and completion)
- Code block and structured data detection
- Reasoning keyword signals ("analyze", "compare", "design", etc.)
- Multi-turn conversation patterns

**3. Cost** — Calculates current spend using built-in pricing tables (13 models across OpenAI, Anthropic, Google, Mistral, Meta).

**4. Migrate** — Maps each complexity tier to a cheaper model:
- `simple` → gpt-4o-mini / claude-3-haiku / llama-3.1-8b
- `medium` → gpt-4o-mini / claude-3.5-haiku / mistral-small
- `complex` → keep on current model

**5. Report** — Generates Markdown or JSON report with cost breakdown and savings projections.

## Sample Report

```
# LLM Usage Analysis Report

## Summary
- **Total queries analyzed:** 1,247
- **Date range:** 2026-01-01 to 2026-03-15
- **Current monthly cost:** $342.18

## Query Distribution
| Complexity | Count | % of Total | Current Cost |
|------------|------:|------------|-------------:|
| Simple     |   743 |      59.6% |      $89.16 |
| Medium     |   389 |      31.2% |     $155.60 |
| Complex    |   115 |       9.2% |      $97.42 |

## Projected Savings
- **Current monthly cost:** $342.18
- **Projected monthly cost:** $56.48
- **Monthly savings:** $285.70 (83.5%)
- **Annual savings:** $3,428.40
```

## Library Usage

```python
from llm_cost_router.analyzer import analyze
from llm_cost_router.report_generator import generate_markdown_report

report = analyze("usage_logs.json")
print(f"Potential savings: ${report.total_savings:.2f}/month ({report.savings_pct}%)")
print(generate_markdown_report(report))
```

## Built-in Pricing

Covers 13 models from OpenAI, Anthropic, Google, Mistral, and Meta (self-hosted). Override with custom pricing:

```bash
llm-router pricing --update custom_pricing.json
```

Custom pricing JSON format:
```json
{
  "my-custom-model": {"input": 1.00, "output": 3.00, "provider": "custom"}
}
```

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `LLM_ROUTER_LOG_LEVEL` | `WARNING` | Log level (DEBUG, INFO, WARNING, ERROR) |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

## License

MIT
