"""
data_utils.py
-------------
Helpers for loading, validating, combining and exporting LifeStat data.

Nothing in this file uses Streamlit, so it can be tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "data" / "sample_lifestyle_data.csv"

NUMERIC_COLUMNS = [
    "sleep_hours",
    "exercise_minutes",
    "water_litres",
    "screen_hours",
    "study_hours",
    "energy",
]
REQUIRED_COLUMNS = ["date"] + NUMERIC_COLUMNS
OPTIONAL_COLUMNS = ["mood", "sleep_quality"]
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Allowed (minimum, maximum) for each numeric column.
VALID_RANGES = {
    "sleep_hours": (0.0, 24.0),
    "exercise_minutes": (0.0, 1440.0),
    "water_litres": (0.0, 10.0),
    "screen_hours": (0.0, 24.0),
    "study_hours": (0.0, 24.0),
    "energy": (1.0, 10.0),
}

# Friendly names (with units) used in the app.
LABELS = {
    "sleep_hours": "Sleep duration (hours)",
    "exercise_minutes": "Exercise duration (minutes)",
    "water_litres": "Water intake (litres)",
    "screen_hours": "Screen time (hours)",
    "study_hours": "Study/work hours",
    "energy": "Energy rating (1-10)",
}

# Short names used inside sentences and equations.
SHORT_NAMES = {
    "sleep_hours": "Sleep",
    "exercise_minutes": "Exercise",
    "water_litres": "Water",
    "screen_hours": "Screen time",
    "study_hours": "Study hours",
    "energy": "Energy",
}

MOOD_OPTIONS = ["Happy", "Calm", "Neutral", "Tired", "Stressed", "Sad"]
SLEEP_QUALITY_OPTIONS = ["Poor", "Fair", "Good", "Excellent"]
NOT_RECORDED = "Not recorded"

# Other column names people commonly use in their own spreadsheets.
COLUMN_ALIASES = {
    "sleep": "sleep_hours",
    "sleep_duration": "sleep_hours",
    "sleep_hrs": "sleep_hours",
    "exercise": "exercise_minutes",
    "exercise_duration": "exercise_minutes",
    "exercise_mins": "exercise_minutes",
    "water": "water_litres",
    "water_intake": "water_litres",
    "water_liters": "water_litres",
    "screen_time": "screen_hours",
    "screen": "screen_hours",
    "study": "study_hours",
    "work_hours": "study_hours",
    "study_work_hours": "study_hours",
    "energy_rating": "energy",
    "energy_level": "energy",
    "sleepquality": "sleep_quality",
}

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB is plenty for daily records


@dataclass
class ValidationResult:
    """Outcome of checking a dataset: cleaned data plus messages for the user."""

    data: pd.DataFrame
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and len(self.data) > 0


# ---------------------------------------------------------------------------
# Creating data frames
# ---------------------------------------------------------------------------
def empty_dataframe() -> pd.DataFrame:
    """An empty table that already has all the right columns."""
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            **{c: pd.Series(dtype="float64") for c in NUMERIC_COLUMNS},
            "mood": pd.Series(dtype="object"),
            "sleep_quality": pd.Series(dtype="object"),
        }
    )[ALL_COLUMNS]


def row_issues(values: dict) -> list:
    """
    Check ONE record (a dictionary) and return a list of plain-English problems.
    An empty list means the record is valid.
    """
    problems = []
    if pd.isna(values.get("date")):
        problems.append("The date is missing or not a valid date.")
    for col in NUMERIC_COLUMNS:
        value = pd.to_numeric(values.get(col), errors="coerce")
        low, high = VALID_RANGES[col]
        if pd.isna(value):
            problems.append(f"{LABELS[col]} is missing or is not a number.")
        elif value < low or value > high:
            problems.append(f"{LABELS[col]} must be between {low:g} and {high:g}.")
    sleep = pd.to_numeric(values.get("sleep_hours"), errors="coerce")
    study = pd.to_numeric(values.get("study_hours"), errors="coerce")
    if pd.notna(sleep) and pd.notna(study) and sleep + study > 24:
        problems.append("Sleep hours and study/work hours together cannot be more than 24.")
    return problems


def make_record_frame(values: dict) -> pd.DataFrame:
    """Turn one valid record (dictionary) into a one-row table."""
    row = {
        "date": pd.Timestamp(values["date"]).normalize(),
        **{c: float(values[c]) for c in NUMERIC_COLUMNS},
        "mood": values.get("mood") or NOT_RECORDED,
        "sleep_quality": values.get("sleep_quality") or NOT_RECORDED,
    }
    return pd.DataFrame([row])[ALL_COLUMNS]


def combine_records(existing: pd.DataFrame, new: pd.DataFrame):
    """
    Add new records to existing ones. If a date already exists, the new record
    replaces the old one. Returns (combined table, number of replaced dates).
    """
    if existing is None or len(existing) == 0:
        return new.sort_values("date").reset_index(drop=True), 0
    combined = pd.concat([existing, new], ignore_index=True)
    replaced = int(combined.duplicated("date", keep="last").sum())
    combined = (
        combined.drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return combined, replaced


# ---------------------------------------------------------------------------
# Validating an uploaded table
# ---------------------------------------------------------------------------
def _clean_name(name) -> str:
    cleaned = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return COLUMN_ALIASES.get(cleaned, cleaned)


def validate_dataframe(raw: pd.DataFrame) -> ValidationResult:
    """
    Check an uploaded table. Missing columns are errors. Bad rows are removed
    and reported as warnings, so one typo does not reject a whole file.
    """
    errors, warnings = [], []

    if raw is None or raw.empty:
        return ValidationResult(empty_dataframe(), ["The file has no data rows."], [])

    df = raw.copy()
    df.columns = [_clean_name(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(
            "These required columns are missing: "
            + ", ".join(missing)
            + ". Please download the sample template and use the same column names."
        )
        return ValidationResult(empty_dataframe(), errors, warnings)

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = NOT_RECORDED
    df = df[ALL_COLUMNS]

    # Convert types. Anything that cannot be converted becomes missing (NaN/NaT).
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Find invalid rows.
    bad = df["date"].isna()
    for col in NUMERIC_COLUMNS:
        low, high = VALID_RANGES[col]
        bad = bad | df[col].isna() | (df[col] < low) | (df[col] > high)
    bad = bad | ((df["sleep_hours"] + df["study_hours"]) > 24)

    if bad.any():
        shown = 0
        for idx in df.index[bad]:
            if shown >= 8:
                break
            problems = row_issues(df.loc[idx].to_dict())
            warnings.append(f"Row {idx + 1} was skipped: " + " ".join(problems))
            shown += 1
        total_bad = int(bad.sum())
        if total_bad > shown:
            warnings.append(f"...and {total_bad - shown} more invalid rows were skipped.")
        df = df[~bad]

    if df.empty:
        errors.append("No valid rows were found, so nothing can be analysed.")
        return ValidationResult(empty_dataframe(), errors, warnings)

    # Energy is a whole-number rating.
    if (df["energy"] % 1 != 0).any():
        df["energy"] = df["energy"].round()
        warnings.append("Some energy ratings had decimals and were rounded to whole numbers.")

    # Optional text columns.
    for col in OPTIONAL_COLUMNS:
        cleaned = df[col].fillna("").astype(str).str.strip().str.title()
        cleaned = cleaned.replace({"": NOT_RECORDED, "Nan": NOT_RECORDED, "None": NOT_RECORDED})
        df[col] = cleaned.replace({"Not Recorded": NOT_RECORDED})

    # Duplicate dates: keep the last record for each date.
    duplicate_count = int(df.duplicated("date", keep="last").sum())
    if duplicate_count:
        df = df.drop_duplicates("date", keep="last")
        warnings.append(
            f"{duplicate_count} duplicate date(s) found. Only the last record for each date was kept."
        )

    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 5:
        warnings.append("Fewer than 5 valid records: some analyses need at least 5 records.")
    return ValidationResult(df, errors, warnings)


def load_sample() -> pd.DataFrame:
    """Load the built-in illustrative sample dataset."""
    if not SAMPLE_PATH.exists():
        return empty_dataframe()
    result = validate_dataframe(pd.read_csv(SAMPLE_PATH))
    return result.data


# ---------------------------------------------------------------------------
# Exporting
# ---------------------------------------------------------------------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert records to CSV bytes (dates written as YYYY-MM-DD)."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False).encode("utf-8")


def template_csv_bytes() -> bytes:
    """A small example file users can copy the format from."""
    example = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "sleep_hours": [7.5, 6.0, 8.0],
            "exercise_minutes": [30, 0, 45],
            "water_litres": [2.2, 1.6, 2.5],
            "screen_hours": [5.0, 7.5, 4.0],
            "study_hours": [4.0, 5.5, 3.0],
            "energy": [7, 5, 8],
            "mood": ["Happy", "Tired", "Calm"],
            "sleep_quality": ["Good", "Fair", "Excellent"],
        }
    )
    return example.to_csv(index=False).encode("utf-8")


def display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """A friendly-looking copy of the records for showing in tables."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["energy"] = out["energy"].astype(int)
    out = out.rename(
        columns={**LABELS, "date": "Date", "mood": "Mood", "sleep_quality": "Sleep quality"}
    )
    return out
