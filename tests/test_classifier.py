import pytest

from llm_cost_router.classifier import classify_query
from llm_cost_router.models import QueryRecord


@pytest.mark.parametrize(
    "prompt_tokens,expected",
    [
        (30, "simple"),
        (100, "simple"),
        (199, "simple"),
        (500, "medium"),
        (999, "medium"),
        (1500, "complex"),
        (3000, "complex"),
    ],
)
def test_classify_by_token_count_only(prompt_tokens, expected):
    """Token count alone should drive classification when no text is available."""
    rec = QueryRecord(model="gpt-4o", prompt_tokens=prompt_tokens, completion_tokens=50)
    result = classify_query(rec)
    assert result.complexity == expected


def test_simple_query():
    rec = QueryRecord(
        model="gpt-4o",
        prompt_tokens=30,
        completion_tokens=15,
        prompt_text="Translate 'hello' to French",
        response_text="Bonjour",
    )
    result = classify_query(rec)
    assert result.complexity == "simple"
    assert result.confidence >= 0.7
    assert "low_token_count" in result.signals


def test_complex_query_with_code():
    rec = QueryRecord(
        model="gpt-4o",
        prompt_tokens=150,
        completion_tokens=650,
        prompt_text="Write a Python function for binary search",
        response_text=(
            "```python\ndef binary_search(arr, target):\n"
            "    left = 0\n    right = len(arr) - 1\n"
            "    while left <= right:\n        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n            return mid\n```"
        ),
    )
    result = classify_query(rec)
    assert result.complexity in ("medium", "complex")
    assert "code_blocks" in result.signals


def test_complex_query_high_tokens():
    rec = QueryRecord(
        model="gpt-4o",
        prompt_tokens=2000,
        completion_tokens=1500,
        prompt_text="Analyze this entire codebase and suggest refactoring improvements",
    )
    result = classify_query(rec)
    assert result.complexity == "complex"
    assert "high_token_count" in result.signals


def test_medium_query_structured_output():
    rec = QueryRecord(
        model="gpt-4o",
        prompt_tokens=350,
        completion_tokens=200,
        prompt_text="Extract fields from this text and return as JSON",
        response_text='{"name": "John", "email": "john@example.com", "phone": "555-0142"}',
    )
    result = classify_query(rec)
    assert result.complexity == "medium"
    assert "structured_data" in result.signals


def test_reasoning_keywords_increase_complexity():
    rec = QueryRecord(
        model="gpt-4o",
        prompt_tokens=180,
        completion_tokens=300,
        prompt_text="Compare and analyze the trade-off between consistency and availability",
    )
    result = classify_query(rec)
    assert any("reasoning:" in s for s in result.signals)
    assert result.complexity in ("medium", "complex")


def test_simple_task_keywords():
    rec = QueryRecord(
        model="gpt-4o",
        prompt_tokens=40,
        completion_tokens=20,
        prompt_text="Summarize the following paragraph briefly",
    )
    result = classify_query(rec)
    assert result.complexity == "simple"
    assert any("simple_task:" in s for s in result.signals)


def test_confidence_range():
    """All confidence values should be between 0 and 1."""
    test_cases = [
        QueryRecord(model="gpt-4o", prompt_tokens=20, completion_tokens=10),
        QueryRecord(model="gpt-4o", prompt_tokens=500, completion_tokens=300),
        QueryRecord(model="gpt-4o", prompt_tokens=2000, completion_tokens=1500),
    ]
    for rec in test_cases:
        result = classify_query(rec)
        assert 0.0 <= result.confidence <= 1.0


# TODO: add edge case tests for multi-turn conversations with system prompts
