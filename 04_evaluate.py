import pandas as pd
from utils import load_data

df = load_data()
feat = pd.read_csv("data/features.csv", index_col=0)

true_mules = set(df[df["isFraud"] == 1]["receiver"])
flagged = set(feat[feat["flag"]].index)

tp = len(flagged & true_mules)
precision = tp / len(flagged) if flagged else 0
recall = tp / len(true_mules) if true_mules else 0

print(f"Flagged: {len(flagged)} | True mule accounts: {len(true_mules)}")
print(f"Caught: {tp}")
print(f"Precision: {precision:.1%} | Recall: {recall:.1%}")