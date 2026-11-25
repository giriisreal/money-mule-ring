import pandas as pd
import networkx as nx

def load_data():
    return pd.read_csv("data/transactions.csv")

def build_graph(df):
    return nx.from_pandas_edgelist(
        df, "sender", "receiver", ["amount", "hour", "isFraud"],
        create_using=nx.MultiDiGraph())