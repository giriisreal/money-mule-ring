import numpy as np
import pandas as pd
import networkx as nx
from utils import load_data, build_graph

df = load_data()

feat = pd.DataFrame({
    "in":      df.groupby("receiver")["amount"].sum(),
    "out":     df.groupby("sender")["amount"].sum(),
    "fan_in":  df.groupby("receiver")["sender"].nunique(),
    "fan_out": df.groupby("sender")["receiver"].nunique(),
}).fillna(0)

# how much of the money received gets passed on
feat["pass_through"] = (feat["out"] / feat["in"].replace(0, np.nan)).clip(upper=1).fillna(0)

# rule: receives a lot of money and passes most of it on
feat["flag"] = (feat["in"] > 100_000) & (feat["pass_through"] > 0.8)

feat.to_csv("data/features.csv")
print("Flagged accounts:", int(feat["flag"].sum()))

# group flagged accounts into rings (connected clusters)
G = build_graph(df)
flagged = set(feat[feat["flag"]].index)
H = nx.Graph(G.subgraph(flagged))
rings = [c for c in nx.connected_components(H) if len(c) >= 2]
print("Possible rings (2+ linked flagged accounts):", len(rings))