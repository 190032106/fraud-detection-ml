"""Predict whether one transaction is potentially fraudulent."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from preprocess import load_model, prepare_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model/fraud_model.pkl"))
    parser.add_argument("--time", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--sender-account", required=True)
    parser.add_argument("--receiver-account", required=True)
    parser.add_argument("--amount", type=float, required=True)
    parser.add_argument("--payment-currency", required=True)
    parser.add_argument("--received-currency", required=True)
    parser.add_argument("--sender-bank-location", required=True)
    parser.add_argument("--receiver-bank-location", required=True)
    parser.add_argument("--payment-type", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    model, feature_columns = load_model(args.model)
    transaction = pd.DataFrame([{
        "Time": args.time,
        "Date": args.date,
        "Sender_account": args.sender_account,
        "Receiver_account": args.receiver_account,
        "Amount": args.amount,
        "Payment_currency": args.payment_currency,
        "Received_currency": args.received_currency,
        "Sender_bank_location": args.sender_bank_location,
        "Receiver_bank_location": args.receiver_bank_location,
        "Payment_type": args.payment_type,
    }])
    probability = float(model.predict_proba(prepare_features(transaction, feature_columns))[:, 1][0])
    label = "FRAUD REVIEW" if probability >= args.threshold else "NORMAL"
    print(f"Result: {label}")
    print(f"Fraud probability: {probability:.4f}")


if __name__ == "__main__":
    main(parse_args())