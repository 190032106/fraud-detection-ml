"""Train a fraud classifier from the large CSV in bounded memory."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from preprocess import TARGET_COLUMN, prepare_features


def collect_training_data(csv_path: Path, chunk_size: int, negative_ratio: int, seed: int) -> pd.DataFrame:
    """Keep every fraud row and a bounded random sample of normal rows."""
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        fraud = chunk[chunk[TARGET_COLUMN] == 1]
        normal = chunk[chunk[TARGET_COLUMN] == 0]
        keep = min(len(normal), max(len(fraud) * negative_ratio, 1000))
        if keep:
            normal = normal.sample(n=keep, random_state=seed)
        chunks.append(pd.concat([fraud, normal], ignore_index=True))

    if not chunks:
        raise ValueError("The CSV contains no rows")
    data = pd.concat(chunks, ignore_index=True)
    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("The training data must contain both fraud and normal transactions")
    return data.sample(frac=1, random_state=seed).reset_index(drop=True)


def train(args: argparse.Namespace) -> None:
    data = collect_training_data(args.csv, args.chunk_size, args.negative_ratio, args.seed)
    x = prepare_features(data)
    y = data[TARGET_COLUMN].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)

    print(f"Training rows: {len(x_train):,}; test rows: {len(x_test):,}")
    print(f"Fraud rows in sampled data: {int(y.sum()):,}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, probabilities):.4f}")
    print(classification_report(y_test, predictions, target_names=["normal", "fraud"], zero_division=0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Saved model to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/fraud.csv"))
    parser.add_argument("--output", type=Path, default=Path("model/fraud_model.pkl"))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--trees", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())