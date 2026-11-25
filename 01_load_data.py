import os
import kagglehub
import pandas as pd

path = kagglehub.dataset_download("mtalaltariq/paysim-data")
csv = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv))

df = df.rename(columns={"nameOrig": "sender", "nameDest": "receiver", "step": "hour"})
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]

fraud = df[df["isFraud"] == 1]
normal = df[df["isFraud"] == 0].sample(100_000, random_state=1)
data = pd.concat([fraud, normal]).sort_values("hour")

os.makedirs("data", exist_ok=True)
data.to_csv("data/transactions.csv", index=False)
print("Saved", len(data), "rows,", len(fraud), "fraud")