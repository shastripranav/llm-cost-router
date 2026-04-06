import pandas as pd

from ..models import QueryRecord

REQUIRED_COLUMNS = {"model", "prompt_tokens", "completion_tokens"}


def parse_csv_logs(filepath: str) -> list[QueryRecord]:
    """Parse a CSV log file into QueryRecord objects.

    Args:
        filepath: Path to CSV file with at minimum: model, prompt_tokens,
                  completion_tokens columns. Optional: timestamp, prompt, response.

    Returns:
        List of parsed QueryRecord objects.

    Raises:
        ValueError: If required columns are missing from the CSV.
    """
    # pandas handles mixed dtypes better than csv.DictReader for large files
    df = pd.read_csv(filepath)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    records = []
    for _, row in df.iterrows():
        record = QueryRecord(
            timestamp=_safe_timestamp(row.get("timestamp")),
            model=str(row["model"]),
            prompt_tokens=int(row["prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            prompt_text=_safe_str(row.get("prompt")),
            response_text=_safe_str(row.get("response")),
        )
        records.append(record)

    return records


def _safe_timestamp(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return pd.to_datetime(val).to_pydatetime()
    except Exception:
        return None


def _safe_str(val) -> str | None:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    return str(val)
