import pandas as pd
from sklearn.model_selection import train_test_split


def rc_strata(df: pd.DataFrame) -> pd.Series:
    breaths = df.groupby("breath_id")[["R", "C"]].first()
    return breaths["R"].astype(str) + "_" + breaths["C"].astype(str)


def split_breaths(
    df: pd.DataFrame, strata: pd.Series, val_frac: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    breath_ids = df["breath_id"].unique()
    _, val_ids = train_test_split(
        breath_ids,
        test_size=val_frac,
        stratify=strata.loc[breath_ids],
        random_state=seed,
    )
    is_val = df["breath_id"].isin(val_ids)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
