# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-08

### Added

- Built-in pricing data for 13 LLM models across OpenAI, Anthropic, Google, Mistral, and Meta, with support for custom pricing overrides
- Cost calculation engine that processes usage logs and produces per-model and aggregated breakdowns
- Migration planner that maps each query to a cheaper alternative and reports projected monthly and annual savings
- Heuristic query classifier that labels queries as `simple`, `medium`, or `complex` using token counts, code/structured-data detection, reasoning-keyword signals, and multi-turn patterns
- Log parsers for OpenAI Chat Completions JSON, Anthropic Messages API JSON, and a generic CSV format with auto-detection and manual `--format` override
- CLI (`llm-router`) with subcommands: `pricing` (view/update rates), `analyze` (full report), `savings` (quick estimate), `classify` (per-query complexity output), and `migrate` (migration plan with custom target models)
- Markdown and JSON report generation summarizing query distribution, cost breakdown, and savings projections
- Python library API exposing `analyze()` and `generate_markdown_report()` for programmatic use
- Python 3.10+ compatibility (matrix-tested against 3.10, 3.11, and 3.12 in CI)
