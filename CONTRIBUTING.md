# Contributing to llm-cost-router

This project is MIT licensed and welcomes contributions. The most useful contributions right now are: pricing updates as providers change rates, new provider parsers, and routing strategy improvements.

## How to Contribute

1. Fork the repository.
2. Branch off `main` with a descriptive name: `git checkout -b feat/cohere-parser`.
3. Make your changes, run tests and the formatter, and confirm everything is green.
4. Open a pull request describing the change and any pricing/routing implications.

## Development setup

```bash
pip install -e ".[dev]"
```

## Code style

The codebase uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting. CI enforces both:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

If `ruff format --check` complains, run `ruff format src/ tests/` to apply.

## Testing

```bash
pytest -v
```

The CI matrix runs against Python 3.10, 3.11, and 3.12 — please make sure your changes work on all three. If you're updating pricing, please also update the `PRICING_LAST_UPDATED` constant.

## Questions

Open an issue with the `question` label.
