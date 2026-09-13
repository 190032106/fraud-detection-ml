"""Feature preparation shared by training and prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


RAW_COLUMNS = [
    "Time",
    "Date",
    "Sender_account",
    "Receiver_account",
    "Amount",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
]
TARGET_COLUMN = "Is_laundering"


def prepare_features(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Convert raw transactions into the numeric feature matrix used by the model."""
    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    features = frame[RAW_COLUMNS].copy()
    timestamp = pd.to_datetime(
        features["Date"].astype(str) + " " + features["Time"].astype(str),
        errors="coerce",
    )
    if timestamp.isna().any():
        raise ValueError("Time and Date contain invalid values")

    features["Amount"] = pd.to_numeric(features["Amount"], errors="raise")
    features["hour"] = timestamp.dt.hour
    features["day"] = timestamp.dt.day
    features = features.drop(columns=["Time", "Date", "Sender_account", "Receiver_account"])
    features = pd.get_dummies(features, dtype=float)

    if feature_columns is not None:
        features = features.reindex(columns=feature_columns, fill_value=0.0)
    return features.astype(float)


def load_model(model_path: str | Path):
    """Load a joblib model and return it with its feature names."""
    model = joblib.load(model_path)
    feature_columns = list(getattr(model, "feature_names_in_", []))
    if not feature_columns:
        raise ValueError("The model does not contain feature_names_in_")
    return model, feature_columns