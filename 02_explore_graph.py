import networkx as nx
from utils import load_data, build_graph

df = load_data()
G = build_graph(df)

print("Accounts:", G.number_of_nodes())
print("Transactions:", G.number_of_edges())
print("Weakly connected groups:", nx.number_weakly_connected_components(G))

top = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:5]
print("Accounts receiving from the most senders:", top)