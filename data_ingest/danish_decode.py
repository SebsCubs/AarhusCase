"""Fix Latin-1 / UTF-8 mojibake in Danish column names and replace '--' with NaN."""
import re
import pandas as pd


_REPLACEMENTS = {
    "Â°": "°",
    "Ã¸": "ø",
    "Ã¦": "æ",
    "Ã…": "Å",
    "Ã¥": "å",
    "Ã†": "Æ",
    "Ã˜": "Ø",
}


def fix_string(s: str) -> str:
    for bad, good in _REPLACEMENTS.items():
        s = s.replace(bad, good)
    return s


def fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [fix_string(c) for c in df.columns]
    return df


def replace_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '--' and empty strings with NaN."""
    return df.replace({"--": pd.NA, "": pd.NA})
