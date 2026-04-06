from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_openai_path():
    return str(FIXTURES_DIR / "sample_openai_logs.json")


@pytest.fixture
def sample_csv_path():
    return str(FIXTURES_DIR / "sample_usage.csv")
