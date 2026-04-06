import pytest

from llm_cost_router.cost_engine import aggregate_costs, calculate_query_cost
from llm_cost_router.models import QueryRecord
from llm_cost_router.pricing import PROVIDER_PRICING


@pytest.mark.parametrize(
    "model,input_tok,output_tok,expected_min,expected_max",
    [
        ("gpt-4o", 1000, 500, 0.007, 0.008),
        ("gpt-4o-mini", 1000, 500, 0.0004, 0.0005),
        ("gpt-4-turbo", 1000, 500, 0.024, 0.026),
        ("claude-3.5-sonnet", 1000, 500, 0.010, 0.011),
        ("llama-3.1-8b", 1000, 500, 0.0, 0.0001),
    ],
)
def test_calculate_query_cost(model, input_tok, output_tok, expected_min, expected_max):
    cost = calculate_query_cost(model, input_tok, output_tok, PROVIDER_PRICING)
    assert expected_min <= cost <= expected_max, f"Got {cost} for {model}"


def test_unknown_model_returns_zero():
    cost = calculate_query_cost("nonexistent-model", 1000, 500)
    assert cost == 0.0


def test_zero_tokens():
    cost = calculate_query_cost("gpt-4o", 0, 0)
    assert cost == 0.0


def test_aggregate_costs_single_model():
    records = [
        QueryRecord(model="gpt-4o", prompt_tokens=1000, completion_tokens=500),
        QueryRecord(model="gpt-4o", prompt_tokens=2000, completion_tokens=1000),
    ]
    estimates = aggregate_costs(records, PROVIDER_PRICING)
    assert len(estimates) == 1
    assert estimates[0].model == "gpt-4o"
    assert estimates[0].total_queries == 2
    assert estimates[0].total_input_tokens == 3000
    assert estimates[0].total_output_tokens == 1500


def test_aggregate_costs_multiple_models():
    records = [
        QueryRecord(model="gpt-4o", prompt_tokens=1000, completion_tokens=500),
        QueryRecord(model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=500),
        QueryRecord(model="gpt-4o", prompt_tokens=500, completion_tokens=200),
    ]
    estimates = aggregate_costs(records, PROVIDER_PRICING)
    assert len(estimates) == 2
    models = [e.model for e in estimates]
    assert "gpt-4o" in models
    assert "gpt-4o-mini" in models

    gpt4o = next(e for e in estimates if e.model == "gpt-4o")
    assert gpt4o.total_queries == 2


def test_aggregate_costs_empty():
    estimates = aggregate_costs([])
    assert estimates == []


def test_avg_cost_per_query():
    records = [
        QueryRecord(model="gpt-4o", prompt_tokens=1000, completion_tokens=500),
        QueryRecord(model="gpt-4o", prompt_tokens=1000, completion_tokens=500),
    ]
    estimates = aggregate_costs(records, PROVIDER_PRICING)
    est = estimates[0]
    assert est.avg_cost_per_query == pytest.approx(est.total_cost / 2, abs=1e-6)
