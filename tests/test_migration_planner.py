import pytest

from llm_cost_router.migration_planner import (
    BUDGET_MODELS,
    generate_migration_plan,
    recommend_target,
)
from llm_cost_router.models import QueryRecord
from llm_cost_router.pricing import PROVIDER_PRICING


class TestRecommendTarget:
    def test_simple_gets_cheaper_model(self):
        target = recommend_target("gpt-4o", "simple")
        assert target == "gpt-4o-mini"

    def test_complex_keeps_current(self):
        target = recommend_target("gpt-4o", "complex")
        assert target == "gpt-4o"

    def test_complex_keeps_any_model(self):
        target = recommend_target("claude-3.5-sonnet", "complex")
        assert target == "claude-3.5-sonnet"

    def test_budget_model_not_migrated(self):
        for model in BUDGET_MODELS:
            target = recommend_target(model, "simple")
            assert target == model, f"{model} should stay put"

    def test_custom_override(self):
        target = recommend_target("gpt-4o", "simple", overrides={"simple": "gemini-1.5-flash"})
        assert target == "gemini-1.5-flash"

    def test_medium_target(self):
        target = recommend_target("gpt-4o", "medium")
        assert target == "gpt-4o-mini"

    def test_medium_override(self):
        target = recommend_target(
            "gpt-4o", "medium", overrides={"medium": "mistral-small"}
        )
        assert target == "mistral-small"


class TestGenerateMigrationPlan:
    @pytest.fixture
    def sample_records(self):
        return [
            QueryRecord(
                model="gpt-4o", prompt_tokens=50,
                completion_tokens=20, complexity="simple",
            ),
            QueryRecord(
                model="gpt-4o", prompt_tokens=60,
                completion_tokens=30, complexity="simple",
            ),
            QueryRecord(
                model="gpt-4o", prompt_tokens=400,
                completion_tokens=200, complexity="medium",
            ),
            QueryRecord(
                model="gpt-4o", prompt_tokens=1500,
                completion_tokens=800, complexity="complex",
            ),
            QueryRecord(
                model="gpt-4-turbo", prompt_tokens=100,
                completion_tokens=50, complexity="simple",
            ),
        ]

    def test_plan_has_correct_query_count(self, sample_records):
        report = generate_migration_plan(sample_records, pricing=PROVIDER_PRICING)
        assert report.total_queries == 5

    def test_plan_distribution(self, sample_records):
        report = generate_migration_plan(sample_records, pricing=PROVIDER_PRICING)
        assert report.distribution["simple"] == 3
        assert report.distribution["medium"] == 1
        assert report.distribution["complex"] == 1

    def test_savings_are_positive(self, sample_records):
        report = generate_migration_plan(sample_records, pricing=PROVIDER_PRICING)
        assert report.total_savings >= 0
        assert report.savings_pct >= 0

    def test_projected_cost_less_than_current(self, sample_records):
        report = generate_migration_plan(sample_records, pricing=PROVIDER_PRICING)
        assert report.projected_total_cost <= report.current_total_cost

    def test_migration_targets_present(self, sample_records):
        report = generate_migration_plan(sample_records, pricing=PROVIDER_PRICING)
        assert len(report.migration_plan) > 0
        for target in report.migration_plan:
            assert target.query_count > 0

    def test_complex_queries_not_migrated(self, sample_records):
        report = generate_migration_plan(sample_records, pricing=PROVIDER_PRICING)
        complex_targets = [t for t in report.migration_plan if t.complexity == "complex"]
        for t in complex_targets:
            assert t.recommended_model == t.current_model

    def test_custom_target_models(self, sample_records):
        report = generate_migration_plan(
            sample_records,
            pricing=PROVIDER_PRICING,
            target_simple="gemini-1.5-flash",
            target_medium="mistral-small",
        )
        simple_targets = [t for t in report.migration_plan if t.complexity == "simple"]
        for t in simple_targets:
            if t.current_model not in BUDGET_MODELS:
                assert t.recommended_model == "gemini-1.5-flash"
