"""Heuristic query complexity classifier.

Classifies each query as simple/medium/complex based on token counts, code
presence, structured data patterns, and reasoning keyword signals. No LLM
calls needed — pure rule-based classification.
"""

import re

from .models import ClassificationResult, QueryRecord

REASONING_KEYWORDS = frozenset(
    {
        "analyze",
        "compare",
        "design",
        "debug",
        "evaluate",
        "architect",
        "optimize",
        "refactor",
        "implement",
        "trade-off",
        "trade off",
        "pros and cons",
        "step by step",
        "reason through",
        "explain why",
    }
)

SIMPLE_TASK_KEYWORDS = frozenset(
    {
        "translate",
        "summarize",
        "classify",
        "extract",
        "list",
        "convert",
        "format",
        "rewrite",
        "fix grammar",
        "define",
        "what is",
        "how many",
        "name the",
    }
)

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_CODE_INDICATORS = re.compile(
    r"\b(def |class |function\s*\(|const |import |require\(|from \w+ import|"
    r"async def |if __name__|console\.log|System\.out)"
)
_JSON_STRUCTURE_RE = re.compile(r"\{[^}]{20,}\}")
_MULTILINE_THRESHOLD = 3


def classify_query(
    record: QueryRecord,
    simple_threshold: int = 200,
    complex_threshold: int = 1000,
) -> ClassificationResult:
    """Classify a query record into simple/medium/complex.

    Args:
        record: The query record to classify.
        simple_threshold: Max prompt tokens to consider as potentially simple.
        complex_threshold: Min prompt tokens that auto-flag as complex.

    Returns:
        ClassificationResult with complexity level, confidence, and triggered signals.
    """
    signals: list[str] = []
    score = 0.0

    # --- Token count signals ---
    # thresholds tuned so token count alone is sufficient for clear-cut cases
    if record.prompt_tokens >= complex_threshold:
        signals.append("high_token_count")
        score += 4.5
    elif record.prompt_tokens >= simple_threshold:
        signals.append("moderate_token_count")
        score += 2.0
    else:
        signals.append("low_token_count")

    if record.completion_tokens > 800:
        signals.append("long_response")
        score += 1.5

    combined = f"{record.prompt_text or ''}\n{record.response_text or ''}"
    prompt_lower = (record.prompt_text or "").lower()

    # --- Code detection ---
    if _CODE_BLOCK_RE.search(combined):
        signals.append("code_blocks")
        score += 2.5
    elif _CODE_INDICATORS.search(combined):
        signals.append("code_indicators")
        score += 1.5

    # --- Structured data ---
    if _JSON_STRUCTURE_RE.search(combined):
        signals.append("structured_data")
        score += 1.0

    # --- Reasoning keywords ---
    reasoning_hits = [kw for kw in REASONING_KEYWORDS if kw in prompt_lower]
    if reasoning_hits:
        signals.append(f"reasoning:{','.join(reasoning_hits[:3])}")
        # TODO: add weighted confidence scoring based on signal reliability
        score += min(len(reasoning_hits) * 0.8, 2.5)

    # --- Simple task patterns ---
    simple_hits = [kw for kw in SIMPLE_TASK_KEYWORDS if kw in prompt_lower]
    if simple_hits and score < 2.0:
        signals.append(f"simple_task:{','.join(simple_hits[:2])}")
        score = max(0, score - 0.5)

    # --- Multi-turn detection (newlines in prompt suggest multiple messages) ---
    if record.prompt_text and record.prompt_text.count("\n") >= _MULTILINE_THRESHOLD:
        newline_count = record.prompt_text.count("\n")
        if newline_count > 8:
            signals.append("multi_turn_context")
            score += 1.5

    return _score_to_result(score, signals)


def _score_to_result(score: float, signals: list[str]) -> ClassificationResult:
    if score <= 1.5:
        complexity = "simple"
        conf = min(0.92, 0.70 + (1.5 - score) * 0.15)
    elif score <= 4.0:
        complexity = "medium"
        conf = 0.60 + (score - 1.5) * 0.08
    else:
        complexity = "complex"
        conf = min(0.95, 0.65 + (score - 4.0) * 0.06)

    return ClassificationResult(
        complexity=complexity,
        confidence=round(conf, 2),
        signals=signals,
    )
