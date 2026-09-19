import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.split import rc_strata, split_breaths


def load_data(path: str) -> pd.DataFrame:
    return (
        pd.read_csv(path).sort_values(["breath_id", "time_step"]).reset_index(drop=True)
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    for k in range(1, 5):
        df[f"u_in_lag{k}"] = df.groupby("breath_id")["u_in"].shift(k).fillna(0)
        df[f"u_in_lead{k}"] = df.groupby("breath_id")["u_in"].shift(-k).fillna(0)
        df[f"u_in_diff{k}"] = df["u_in"] - df[f"u_in_lag{k}"].fillna(0)
        df[f"u_in_diff_lead{k}"] = df[f"u_in_lead{k}"] - df["u_in"].fillna(0)

    df["u_in_max"] = df.groupby("breath_id")["u_in"].transform("max")

    df["R_C"] = df["R"].astype(str) + "_" + df["C"].astype(str)
    df = pd.get_dummies(df, columns=["R_C"]).drop(columns=["R", "C"])
    return df


NON_FEATURES = ["id", "breath_id", "pressure"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


# compute only on training data
def compute_norm_stats(df: pd.DataFrame, feature_cols, skip_cols) -> tuple:
    mean = df[feature_cols].mean()
    std = df[feature_cols].std()
    mean[skip_cols] = 0.0
    std[skip_cols] = 1.0
    return mean, std


STEPS = 80 


class BreathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, mean, std):
        n = df["breath_id"].nunique()
        features = ((df[feature_cols] - mean) / std).to_numpy(np.float32, copy=True)
        self.features = torch.from_numpy(features).view(n, STEPS, -1)
        self.mask = torch.tensor((df["u_out"] == 0).to_numpy()).view(n, STEPS)
        self.target = (
            torch.from_numpy(df["pressure"].to_numpy(np.float32, copy=True)).view(n, STEPS)
            if "pressure" in df.columns
            else None
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        target = self.target[idx] if self.target is not None else None
        return self.features[idx], self.mask[idx], target


class BreathDatasetCollator:
    def __call__(self, batch):
        features, mask, target = zip(*batch)

        batch = {
            "features": torch.stack(features),
            "mask": torch.stack(mask),
        }
        if target[0] is not None:
            batch["target"] = torch.stack(target)
        return batch


def make_loaders(path: str, batch_size: int = 256, val_batch_size: int = 1024):
    raw = load_data(path)
    strata = rc_strata(raw) 
    df = add_features(raw)
    train_part, val_part = split_breaths(df, strata)

    feature_cols = feature_columns(df)
    one_hot = [c for c in feature_cols if c.startswith("R_C_")]  
    mean, std = compute_norm_stats(train_part, feature_cols, one_hot)

    collator = BreathDatasetCollator()
    train_loader = DataLoader(
        BreathDataset(train_part, feature_cols, mean, std),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        BreathDataset(val_part, feature_cols, mean, std),
        batch_size=val_batch_size,
        collate_fn=collator,
    )
    info = {"feature_cols": feature_cols, "mean": mean, "std": std}
    return train_loader, val_loader, info
