import json

import pytest

from llm_cost_router.parsers import detect_and_parse, detect_format
from llm_cost_router.parsers.anthropic_parser import parse_anthropic_logs
from llm_cost_router.parsers.csv_parser import parse_csv_logs
from llm_cost_router.parsers.openai_parser import parse_openai_logs


class TestOpenAIParser:
    def test_parse_valid_logs(self, sample_openai_path):
        records = parse_openai_logs(sample_openai_path)
        assert len(records) == 50
        assert records[0].model == "gpt-4o"
        assert records[0].prompt_tokens == 32

    def test_parse_extracts_prompt_text(self, sample_openai_path):
        records = parse_openai_logs(sample_openai_path)
        assert records[0].prompt_text is not None
        assert "Translate" in records[0].prompt_text

    def test_parse_extracts_response_text(self, sample_openai_path):
        records = parse_openai_logs(sample_openai_path)
        assert records[0].response_text is not None

    def test_parse_handles_no_messages(self, sample_openai_path):
        records = parse_openai_logs(sample_openai_path)
        record_11 = records[10]
        assert record_11.prompt_text is None
        assert record_11.response_text is None
        assert record_11.prompt_tokens == 55

    def test_parse_single_entry(self, tmp_path):
        entry = {
            "model": "gpt-4o",
            "created": 1767657600,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        path = tmp_path / "single.json"
        path.write_text(json.dumps(entry))

        records = parse_openai_logs(str(path))
        assert len(records) == 1
        assert records[0].model == "gpt-4o"

    def test_parse_skips_malformed(self, tmp_path):
        data = [
            {"model": "gpt-4o", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            {"bad_key": "no model field"},
        ]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(data))

        records = parse_openai_logs(str(path))
        assert len(records) == 1


class TestCSVParser:
    def test_parse_valid_csv(self, sample_csv_path):
        records = parse_csv_logs(sample_csv_path)
        assert len(records) == 20
        assert records[0].model == "gpt-4o"
        assert records[0].prompt_tokens == 25

    def test_parse_csv_with_prompt(self, sample_csv_path):
        records = parse_csv_logs(sample_csv_path)
        assert records[0].prompt_text == "Translate hello to Spanish"
        assert records[0].response_text == "Hola"

    def test_parse_csv_null_prompt(self, sample_csv_path):
        records = parse_csv_logs(sample_csv_path)
        no_text = [r for r in records if r.prompt_text is None]
        assert len(no_text) > 0

    def test_parse_csv_missing_required_columns(self, tmp_path):
        csv_content = "timestamp,model\n2026-01-01,gpt-4o\n"
        path = tmp_path / "bad.csv"
        path.write_text(csv_content)

        with pytest.raises(ValueError, match="missing required columns"):
            parse_csv_logs(str(path))

    def test_parse_csv_without_optional_columns(self, tmp_path):
        csv_content = "model,prompt_tokens,completion_tokens\ngpt-4o,100,50\n"
        path = tmp_path / "minimal.csv"
        path.write_text(csv_content)

        records = parse_csv_logs(str(path))
        assert len(records) == 1
        assert records[0].timestamp is None
        assert records[0].prompt_text is None


class TestAnthropicParser:
    def test_parse_valid_anthropic(self, tmp_path):
        data = [
            {
                "model": "claude-3.5-sonnet",
                "created_at": "2026-01-15T10:00:00Z",
                "usage": {"input_tokens": 200, "output_tokens": 150},
                "messages": [{"role": "user", "content": "Hello"}],
                "content": [{"type": "text", "text": "Hi there!"}],
            },
            {
                "model": "claude-3-haiku",
                "usage": {"input_tokens": 50, "output_tokens": 30},
            },
        ]
        path = tmp_path / "anthropic.json"
        path.write_text(json.dumps(data))

        records = parse_anthropic_logs(str(path))
        assert len(records) == 2
        assert records[0].model == "claude-3.5-sonnet"
        assert records[0].prompt_tokens == 200
        assert records[0].completion_tokens == 150
        assert records[0].prompt_text == "Hello"
        assert records[0].response_text == "Hi there!"

    def test_parse_anthropic_content_blocks(self, tmp_path):
        data = {
            "model": "claude-3.5-sonnet",
            "usage": {"input_tokens": 100, "output_tokens": 80},
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Explain gravity"}]}
            ],
            "content": [
                {"type": "text", "text": "Gravity is a fundamental force."},
                {"type": "text", "text": "It attracts objects with mass."},
            ],
        }
        path = tmp_path / "blocks.json"
        path.write_text(json.dumps(data))

        records = parse_anthropic_logs(str(path))
        assert len(records) == 1
        assert "Gravity" in records[0].response_text


class TestFormatDetection:
    def test_detect_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("model,prompt_tokens,completion_tokens\n")
        assert detect_format(str(path)) == "csv"

    def test_detect_openai_json(self, sample_openai_path):
        assert detect_format(sample_openai_path) == "openai"

    def test_detect_anthropic_json(self, tmp_path):
        data = [{"model": "claude-3-haiku", "usage": {"input_tokens": 50, "output_tokens": 30}}]
        path = tmp_path / "anthropic.json"
        path.write_text(json.dumps(data))
        assert detect_format(str(path)) == "anthropic"

    def test_detect_and_parse_auto(self, sample_openai_path):
        records = detect_and_parse(sample_openai_path)
        assert len(records) == 50

    def test_detect_and_parse_explicit_format(self, sample_csv_path):
        records = detect_and_parse(sample_csv_path, fmt="csv")
        assert len(records) == 20

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            detect_and_parse("test.json", fmt="yaml")
