from pathlib import Path

import pandas as pd

from src.fraud_detection.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN
)

from src.fraud_detection.data import (
    load_data,
    validate_target
)


def test_dataset_exists():
    """Verify that the raw dataset exists."""

    assert RAW_DATA_PATH.exists()


def test_dataset_loads():
    """Verify that the dataset can be loaded."""

    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_target_column_exists():
    """Verify that the target column exists."""

    df = load_data()

    assert TARGET_COLUMN in df.columns


def test_dataset_has_expected_features():
    """Verify the expected credit card dataset structure."""

    df = load_data()

    expected_columns = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
        "Class"
    ]

    assert list(df.columns) == expected_columns


def test_target_is_binary():
    """Verify that Class contains only 0 and 1."""

    df = load_data()

    unique_values = set(
        df[TARGET_COLUMN].dropna().unique()
    )

    assert unique_values.issubset({0, 1})


def test_validate_target():
    """Verify target validation works."""

    df = load_data()

    assert validate_target(df) is True