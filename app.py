import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from utils import load_data, build_graph

st.set_page_config(page_title="Mule Ring Detector", layout="wide")
st.title("Money Mule Ring Detector")


@st.cache_data
def get_data():
    df = load_data()
    feat = pd.DataFrame({
        "in": df.groupby("receiver")["amount"].sum(),
        "out": df.groupby("sender")["amount"].sum(),
        "fan_in": df.groupby("receiver")["sender"].nunique(),
        "fan_out": df.groupby("sender")["receiver"].nunique(),
    }).fillna(0)
    feat["pass_through"] = (
        (feat["out"] / feat["in"].replace(0, np.nan)).clip(upper=1).fillna(0)
    )
    true_mules = set(df[df["isFraud"] == 1]["receiver"])
    return df, feat, true_mules


@st.cache_resource
def get_graph():
    return build_graph(load_data())


df, feat, true_mules = get_data()
G = get_graph()

# ---------- Sidebar: detection rules ----------
st.sidebar.header("Detection rules")
min_in = st.sidebar.slider("Min money received", 0, 1_000_000, 100_000, step=10_000)
min_pt = st.sidebar.slider("Min pass-through ratio", 0.0, 1.0, 0.8, step=0.05)
min_fan = st.sidebar.slider("Min distinct senders", 1, 20, 1)

feat["flag"] = (
    (feat["in"] > min_in)
    & (feat["pass_through"] > min_pt)
    & (feat["fan_in"] >= min_fan)
)
flagged = set(feat[feat["flag"]].index)

# ---------- Metrics ----------
tp = len(flagged & true_mules)
precision = tp / len(flagged) if flagged else 0
recall = tp / len(true_mules) if true_mules else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flagged accounts", len(flagged))
c2.metric("True mule accounts", len(true_mules))
c3.metric("Precision", f"{precision:.1%}")
c4.metric("Recall", f"{recall:.1%}")

# ---------- Flagged table ----------
st.subheader("Flagged accounts")
table = feat[feat["flag"]].sort_values("in", ascending=False)
table = table.assign(confirmed_fraud=table.index.isin(true_mules))
st.dataframe(table[["in", "out", "fan_in", "fan_out",
                    "pass_through", "confirmed_fraud"]].head(200))

# ---------- Rings ----------
H = nx.Graph(G.subgraph(flagged))
rings = [c for c in nx.connected_components(H) if len(c) >= 2]
st.subheader(f"Possible rings: {len(rings)}")

# ---------- Network view ----------
st.subheader("Network view")
if not flagged:
    st.info("No accounts flagged. Loosen the rules in the sidebar.")
else:
    options = list(table.index[:200])
    acct = st.selectbox("Pick a flagged account", options)

    nodes = {acct} | set(G.predecessors(acct)) | set(G.successors(acct))
    nodes = set(list(nodes)[:40])
    nodes.add(acct)
    sub = nx.DiGraph(G.subgraph(nodes))

    colors = []
    for n in sub.nodes:
        if n == acct:
            colors.append("red")
        elif n in true_mules:
            colors.append("purple")
        elif n in flagged:
            colors.append("orange")
        else:
            colors.append("lightblue")

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(sub, seed=1)
    nx.draw(sub, pos, ax=ax, node_color=colors, node_size=300,
            with_labels=False, arrows=True, edge_color="gray")
    st.pyplot(fig)
    st.caption("Red = selected, purple = confirmed fraud, "
               "orange = flagged, blue = other")