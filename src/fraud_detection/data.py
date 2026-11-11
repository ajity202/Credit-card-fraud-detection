import pandas as pd

from .config import RAW_DATA_PATH, TARGET_COLUMN


def load_data(path=RAW_DATA_PATH):
    """
    Load the raw credit card transaction dataset.

    Parameters
    ----------
    path : Path
        Path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    df = pd.read_csv(path)

    return df


def validate_target(df):
    """
    Check whether the target column exists.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to validate.

    Returns
    -------
    bool
        True if target column exists.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found."
        )

    return True