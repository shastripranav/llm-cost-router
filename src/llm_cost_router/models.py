from datetime import datetime

from pydantic import BaseModel, Field


class QueryRecord(BaseModel):
    """Single LLM API call with token usage and optional text content."""

    timestamp: datetime | None = None
    model: str
    prompt_tokens: int
    completion_tokens: int
    prompt_text: str | None = None
    response_text: str | None = None
    complexity: str | None = None


class ClassificationResult(BaseModel):
    complexity: str
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[str]


class CostEstimate(BaseModel):
    """Aggregate cost breakdown for a single model."""

    model: str
    total_queries: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    avg_cost_per_query: float


class MigrationTarget(BaseModel):
    current_model: str
    complexity: str
    recommended_model: str
    query_count: int
    current_cost: float
    projected_cost: float
    savings: float
    savings_pct: float


class AnalysisReport(BaseModel):
    """Full analysis output with cost breakdown and migration recommendations."""

    total_queries: int
    date_range: tuple[str, str] | None = None
    current_total_cost: float
    projected_total_cost: float
    total_savings: float
    savings_pct: float
    distribution: dict[str, int]
    cost_by_model: list[CostEstimate]
    migration_plan: list[MigrationTarget]
