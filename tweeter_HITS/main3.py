#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the graph from a file
def load_graph_from_file(file_path):
    """
    Loads a directed graph from an edge list file.
    Each line in the file should contain two integers: source and target.
    """
    G = nx.DiGraph()
    with open(file_path, "r") as file:
        for line in file:
            source, target = line.strip().split()
            G.add_edge(source, target)
    return G

# Step 2: Apply HITS algorithm
def apply_hits_algorithm(G, max_iter=100, tol=1e-08, top_n=10):
    """
    Runs the HITS algorithm to compute hub and authority scores.
    """
    hubs, authorities = nx.hits(G, max_iter=max_iter, tol=tol, normalized=True)
    top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return hubs, authorities, top_hubs, top_authorities

# Step 3: Detect suspicious users
def detect_suspicious_users(hubs, authorities, hub_percentile=90, authority_percentile=90):
    """
    Detects suspicious nodes with hub and authority scores above specified percentiles.
    """
    hub_scores = np.array(list(hubs.values()))
    authority_scores = np.array(list(authorities.values()))
    hub_threshold = np.percentile(hub_scores, hub_percentile)
    authority_threshold = np.percentile(authority_scores, authority_percentile)
    suspicious_hubs = {node: score for node, score in hubs.items() if score >= hub_threshold}
    suspicious_authorities = {node: score for node, score in authorities.items() if score >= authority_threshold}
    return hub_threshold, authority_threshold, suspicious_hubs, suspicious_authorities

# Step 4: Visualize score distributions with thresholds
def visualize_distributions_with_thresholds(hubs, authorities, hub_threshold, authority_threshold):
    """
    Plots histograms of hub and authority scores with threshold lines.
    """
    plt.figure(figsize=(12, 6))
    plt.hist(hubs.values(), bins=50, alpha=0.7, label="Hub Scores", color="blue")
    plt.axvline(hub_threshold, color="red", linestyle="--", label="Hub Threshold")
    plt.hist(authorities.values(), bins=50, alpha=0.7, label="Authority Scores", color="green")
    plt.axvline(authority_threshold, color="orange", linestyle="--", label="Authority Threshold")
    plt.title("Score Distributions with Thresholds")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Step 5: Visualize top hub and authority nodes
def visualize_top_nodes(G, top_hubs, top_authorities):
    """
    Visualizes the graph with top hub and authority nodes highlighted.
    """
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistent visualization

    # Extract top hub and authority nodes
    top_hub_nodes = {node for node, _ in top_hubs}
    top_authority_nodes = {node for node, _ in top_authorities}
    node_colors = [
        "red" if node in top_hub_nodes else "green" if node in top_authority_nodes else "blue"
        for node in G.nodes()
    ]

    nx.draw(G, pos, node_color=node_colors, node_size=50, edge_color="gray", with_labels=False, alpha=0.7)
    plt.title("Top Nodes (Hubs: Red, Authorities: Green)")
    plt.show()

# Step 6: Visualize suspicious nodes
def visualize_suspicious_nodes(G, suspicious_hubs, suspicious_authorities):
    """
    Visualizes the graph with suspicious hub and authority nodes highlighted.
    """
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistent visualization

    # Extract suspicious hub and authority nodes
    suspicious_hub_nodes = set(suspicious_hubs.keys())
    suspicious_authority_nodes = set(suspicious_authorities.keys())
    node_colors = [
        "purple" if node in suspicious_hub_nodes else "yellow" if node in suspicious_authority_nodes else "gray"
        for node in G.nodes()
    ]

    nx.draw(G, pos, node_color=node_colors, node_size=50, edge_color="gray", with_labels=False, alpha=0.7)
    plt.title("Suspicious Nodes (Hubs: Purple, Authorities: Yellow)")
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Replace with the path to your dataset
    file_path = "twitter_combined.txt"
    
    # Load the graph
    G = load_graph_from_file(file_path)

    # Analyze the largest strongly connected component if necessary
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    if not nx.is_strongly_connected(G):
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest_scc).copy()
        print(f"Reduced graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Ensure there are edges in the graph
    if G.number_of_edges() == 0:
        print("The graph has no edges after reduction. HITS cannot be applied.")
    else:
        # Apply HITS algorithm
        hubs, authorities, top_hubs, top_authorities = apply_hits_algorithm(G)

        # Detect suspicious users with adjusted thresholds
        hub_percentile = 90  # Use the 90th percentile for hubs
        authority_percentile = 90  # Use the 90th percentile for authorities
        hub_threshold, authority_threshold, suspicious_hubs, suspicious_authorities = detect_suspicious_users(
            hubs, authorities, hub_percentile, authority_percentile
        )

        # Display thresholds and suspicious nodes
        print(f"Hub Threshold: {hub_threshold}, Suspicious Hubs: {len(suspicious_hubs)}")
        print(f"Authority Threshold: {authority_threshold}, Suspicious Authorities: {len(suspicious_authorities)}")

        # Visualize score distributions with thresholds
        visualize_distributions_with_thresholds(hubs, authorities, hub_threshold, authority_threshold)

        # Visualize top hub and authority nodes
        visualize_top_nodes(G, top_hubs, top_authorities)

        # Visualize suspicious nodes
        #visualize_suspicious_nodes(G, suspicious_hubs, suspicious_authorities)


# In[ ]:


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def load_graph_from_file(file_path):
    """
    Loads a directed graph from an edge list file.
    Each line in the file should contain two integers: source and target.
    """
    G = nx.DiGraph()
    with open(file_path, "r") as file:
        for line in file:
            source, target = line.strip().split()
            G.add_edge(source, target)
    return G

# Example:
file_path = "twitter_combined.txt"

# Load the graph
G = load_graph_from_file(file_path)


# Step 2: Apply HITS algorithm
def apply_hits_algorithm(G, max_iter=100, tol=1e-08, top_n=10):
    """
    Runs the HITS algorithm and identifies the top hub and authority nodes.
    """
    hubs, authorities = nx.hits(G, max_iter=max_iter, tol=tol, normalized=True)
    top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return hubs, authorities, top_hubs, top_authorities

# Step 3: Detect suspicious users
def detect_suspicious_users(hubs, authorities, hub_percentile=95, authority_percentile=95):
    """
    Detects suspicious nodes with hub and authority scores above specified percentiles.
    """
    hub_scores = np.array(list(hubs.values()))
    authority_scores = np.array(list(authorities.values()))
    hub_threshold = np.percentile(hub_scores, hub_percentile)
    authority_threshold = np.percentile(authority_scores, authority_percentile)
    suspicious_hubs = {node: score for node, score in hubs.items() if score >= hub_threshold}
    suspicious_authorities = {node: score for node, score in authorities.items() if score >= authority_threshold}
    return hub_threshold, authority_threshold, suspicious_hubs, suspicious_authorities

# Step 6: Visualize suspicious nodes
def visualize_suspicious_nodes(G, suspicious_hubs, suspicious_authorities):
    """
    Visualizes the graph with suspicious hub and authority nodes highlighted.
    """
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistent visualization

    # Extract suspicious hub and authority nodes
    suspicious_hub_nodes = set(suspicious_hubs.keys())
    suspicious_authority_nodes = set(suspicious_authorities.keys())
    node_colors = [
        "purple" if node in suspicious_hub_nodes else "yellow" if node in suspicious_authority_nodes else "gray"
        for node in G.nodes()
    ]

    nx.draw(G, pos, node_color=node_colors, node_size=50, edge_color="gray", with_labels=False, alpha=0.7)
    plt.title("Suspicious Nodes (Hubs: Purple, Authorities: Yellow)")
    plt.show()


# Step 5: Visualize overlap
def visualize_overlap(top_hubs, top_authorities, suspicious_hubs, suspicious_authorities):
    """
    Identifies and visualizes overlaps between top nodes and suspicious nodes.
    """
    top_hub_nodes = {node for node, _ in top_hubs}
    top_authority_nodes = {node for node, _ in top_authorities}
    suspicious_hub_nodes = set(suspicious_hubs.keys())
    suspicious_authority_nodes = set(suspicious_authorities.keys())

    overlap_hubs = top_hub_nodes.intersection(suspicious_hub_nodes)
    overlap_authorities = top_authority_nodes.intersection(suspicious_authority_nodes)

    print(f"\nCommon nodes in Top Hub Nodes and Suspicious Hub Nodes: {overlap_hubs}")
    print(f"Common nodes in Top Authority Nodes and Suspicious Authority Nodes: {overlap_authorities}")

# Main Execution
if __name__ == "__main__":


    # Analyze the graph properties
    if not nx.is_strongly_connected(G):
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_wcc).copy()


    # Handle cases with no edges after reduction
    if G.number_of_edges() == 0:
        print("The graph has no edges after reduction. HITS cannot be applied.")
    else:
        # Apply HITS algorithm
        hubs, authorities, top_hubs, top_authorities = apply_hits_algorithm(G)

        # Display top nodes
        print("\nTop Hub Nodes:")
        for node, score in top_hubs:
            print(f"Node: {node}, Hub Score: {score:.6f}")
        
        print("\nTop Authority Nodes:")
        for node, score in top_authorities:
            print(f"Node: {node}, Authority Score: {score:.6f}")


        # Detect suspicious users
        hub_percentile = 90  # Top 10% as suspicious hubs
        authority_percentile = 90  # Top 10% as suspicious authorities
        hub_threshold, authority_threshold, suspicious_hubs, suspicious_authorities = detect_suspicious_users(hubs, authorities, hub_percentile, authority_percentile)



                # Display suspicious nodes
        print("\nSuspicious Hub Nodes (Potential Malicious Users):")
        for node, score in suspicious_hubs.items():
            print(f"Node: {node}, Hub Score: {score:.6f}")
        
        print("\nSuspicious Authority Nodes (Potential Malicious Users):")
        for node, score in suspicious_authorities.items():
            print(f"Node: {node}, Authority Score: {score:.6f}")

        # Visualize overlap
        visualize_overlap(top_hubs, top_authorities, suspicious_hubs, suspicious_authorities)




# In[ ]:


import pickle


def visualize_suspicious_nodes(G, suspicious_hubs, suspicious_authorities, pos=None, pos_file='graph_pos.pkl'):
    """
    Visualizes the graph with suspicious hub and authority nodes highlighted.
    Optimized for large graphs by using a faster layout algorithm and caching positions.
    
    Parameters:
    - G: NetworkX graph
    - suspicious_hubs: dict of suspicious hub nodes
    - suspicious_authorities: dict of suspicious authority nodes
    - pos: Optional precomputed positions (dict)
    - pos_file: Filename to save/load positions
    """
    # Check if positions are provided or cached
    if pos is None:
        if os.path.exists(pos_file):
            print(f"Loading node positions from {pos_file}...")
            with open(pos_file, 'rb') as f:
                pos = pickle.load(f)
        else:
            print("Computing node positions using random_layout...")
            pos = nx.random_layout(G)  # Much faster for large graphs
            with open(pos_file, 'wb') as f:
                pickle.dump(pos, f)
            print(f"Node positions saved to {pos_file}.")

    plt.figure(figsize=(12, 12))
    
    # Extract suspicious hub and authority nodes
    suspicious_hub_nodes = set(suspicious_hubs.keys())
    suspicious_authority_nodes = set(suspicious_authorities.keys())
    
    # Determine node colors
    node_colors = []
    for node in G.nodes():
        if node in suspicious_hub_nodes:
            node_colors.append("purple")
        elif node in suspicious_authority_nodes:
            node_colors.append("yellow")
        else:
            node_colors.append("gray")
    
    # Draw nodes and edges separately for better performance
    # Draw edges first
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", alpha=0.5, width=0.5)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=10,  # Reduced size for performance
        alpha=0.7
    )
    
    plt.title("Suspicious Nodes (Hubs: Purple, Authorities: Yellow)")
    plt.axis('off')  # Hide axes for better visualization
    plt.show()

# Assuming G is your large NetworkX graph
# and suspicious_hubs and suspicious_authorities are dictionaries of nodes

visualize_suspicious_nodes(G, suspicious_hubs, suspicious_authorities)


# In[ ]:


def plot_degree_distribution(G):
    degrees = [degree for _, degree in G.degree()]
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=50, color="blue", alpha=0.7)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

plot_degree_distribution(G)


# In[ ]:


import networkx.algorithms.community as nx_comm

def detect_communities(G):
    communities = nx_comm.greedy_modularity_communities(G)
    print(f"Detected {len(communities)} communities.")
    return communities

# Example:
communities = detect_communities(G)


# In[ ]:


def visualize_communities(G, communities):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    community_colors = plt.cm.get_cmap("tab20", len(communities))
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=list(community), node_color=[community_colors(i)])
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title("Community Visualization")
    plt.show()

# Example:
visualize_communities(G, communities)


# In[ ]:


def centrality_analysis(G):
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    print("\nTop 5 Degree Centrality Nodes:")
    for node, score in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Node: {node}, Score: {score:.4f}")

    print("\nTop 5 Betweenness Centrality Nodes:")
    for node, score in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Node: {node}, Score: {score:.4f}")

    print("\nTop 5 Closeness Centrality Nodes:")
    for node, score in sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Node: {node}, Score: {score:.4f}")

# Example:
centrality_analysis(G)


# In[ ]:


avg_clustering = nx.average_clustering(G.to_undirected())
print(f"Average Clustering Coefficient: {avg_clustering:.4f}")


# In[ ]:


if nx.is_connected(G.to_undirected()):
    diameter = nx.diameter(G.to_undirected())
    print(f"Graph Diameter: {diameter}")


# In[ ]:


def visualize_graph(graph):
    """
    Visualizes a graph using NetworkX and Matplotlib.
    
    Parameters:
        graph (networkx.Graph): The graph to visualize.
    """
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph)  # Compute layout for visualization
    nx.draw_networkx_nodes(graph, pos, node_size=50, node_color="skyblue", alpha=0.7)
    nx.draw_networkx_edges(graph, pos, edge_color="gray", alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="darkgreen")
    
    plt.title("Graph Visualization")
    plt.axis("off")
    plt.show()


subgraph = G.subgraph(list(G.nodes())[:100])  # Example: First 100 nodes
visualize_graph(subgraph)

