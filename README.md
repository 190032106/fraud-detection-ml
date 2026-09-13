# Fraud transaction detection

This project trains a binary classifier for the `Is_laundering` label in `data/fraud.csv`. The CSV is read in chunks because it is large, and normal transactions are sampled at a configurable ratio while all fraud rows are retained.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Train

```powershell
python src/train.py
```

The command prints ROC-AUC, PR-AUC, precision, recall, and F1, then saves the model to `model/fraud_model.pkl`. For fraud data, PR-AUC and recall are more informative than accuracy because fraud is rare.

Useful options include `--negative-ratio 10`, `--trees 300`, and `--threshold 0.3`. Lowering the threshold generally catches more fraud at the cost of more false alerts.

## Predict one transaction

```powershell
python src/predict.py `
  --time 10:35:19 --date 2022-10-07 `
  --sender-account 8724731955 --receiver-account 2769355426 `
  --amount 1459.15 `
  --payment-currency "UK pounds" --received-currency "UK pounds" `
  --sender-bank-location UK --receiver-bank-location UK `
  --payment-type "Cash Deposit"
```