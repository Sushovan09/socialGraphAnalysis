#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# Step 1: Load the Directed Graph from edgelist file
file_path = "congress.edgelist"  # Update with your file's path
graph = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=str)

# Step 2: Compute SimRank Scores
def simrank(G, c=0.8, max_iter=100, eps=1e-4):
    """Calculate SimRank scores for all nodes in a graph."""
    nodes = list(G.nodes())
    n = len(nodes)
    sim = np.identity(n)  # Initialize similarity scores
    node_idx = {node: idx for idx, node in enumerate(nodes)}

    for _ in range(max_iter):
        prev_sim = sim.copy()
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if u == v:
                    continue

                predecessors_u = list(G.predecessors(u))
                predecessors_v = list(G.predecessors(v))

                if not predecessors_u or not predecessors_v:
                    sim[i, j] = 0
                else:
                    sim[i, j] = (c / (len(predecessors_u) * len(predecessors_v))) * sum(
                        prev_sim[node_idx[pre_u], node_idx[pre_v]]
                        for pre_u in predecessors_u
                        for pre_v in predecessors_v
                    )

        # Check convergence
        if np.linalg.norm(sim - prev_sim) < eps:
            break

    return sim, nodes

sim_matrix, node_list = simrank(graph)
sim_df = pd.DataFrame(sim_matrix, index=node_list, columns=node_list)

# Step 3: Display SimRank Scores
print("SimRank Scores (Partial):")
display(sim_df.head())

# Save to a CSV file for further inspection
sim_df.to_csv("simrank_scores.csv")

# Step 4: Visualize SimRank Scores
plt.figure(figsize=(10, 6))
plt.hist(sim_matrix.flatten(), bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of SimRank Scores")
plt.xlabel("SimRank Score")
plt.ylabel("Frequency")
plt.show()

# Step 5: Insights from SimRank Scores
# Find the top-5 most similar node pairs
sim_scores = []
for i, u in enumerate(node_list):
    for j, v in enumerate(node_list):
        if i < j:  # Avoid duplicate pairs
            sim_scores.append((u, v, sim_matrix[i, j]))

sorted_scores = sorted(sim_scores, key=lambda x: -x[2])

print("Top 5 Most Similar Node Pairs:")
display(pd.DataFrame(sorted_scores[:5], columns=["Node1", "Node2", "SimRank Score"]))

# Step 6: Graph Analysis based on SimRank
# Highlight the most similar node pairs on the graph
most_similar_pairs = [(u, v) for u, v, score in sorted_scores[:5]]

pos = nx.spring_layout(graph)
plt.figure(figsize=(12, 8))
nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
nx.draw_networkx_edges(graph, pos, edgelist=most_similar_pairs, edge_color='red', width=2)
plt.title("Graph with Top-5 Similar Node Pairs Highlighted")
plt.show()

# Save the graph with highlighted pairs
nx.write_gexf(graph, "highlighted_graph.gexf")


# In[2]:


# Step 1: Select a subset of nodes and edges for detailed visualization
subset_nodes = set()
for u, v, _ in sorted_scores[:5]:  # Include nodes from the top-5 similar pairs
    subset_nodes.add(u)
    subset_nodes.add(v)

subset_graph = graph.subgraph(subset_nodes)

# Step 2: Visualize the subset graph
subset_pos = nx.spring_layout(subset_graph)
plt.figure(figsize=(8, 6))
nx.draw(subset_graph, subset_pos, with_labels=True, node_color='lightblue', edge_color='gray')
nx.draw_networkx_edges(subset_graph, subset_pos, edgelist=most_similar_pairs, edge_color='red', width=2)
plt.title("Subset Graph with Top-5 Similar Node Pairs Highlighted")
plt.show()


# In[ ]:




