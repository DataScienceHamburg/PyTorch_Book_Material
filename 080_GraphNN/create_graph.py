#%% packages
# datapreparation
from torch_geometric.data import Data
# modeling
import torch
from torch_geometric.utils.convert import to_networkx
# visulalization
import networkx as nx
import matplotlib.pyplot as plt

#%% create the dataset
def create_matchmaking_dataset(
    person_interests: dict
) -> Data:
    # Node features: keep the order of the dict keys for stable indexing
    person_names = list(person_interests.keys())
    x = torch.tensor(list(person_interests.values()), dtype=torch.float)

    # Map names to node indices
    name_to_idx = {name: idx for idx, name in enumerate(person_names)}

    # Build undirected edges as simple pairs (i, j) with i < j
    edges = []

    # Clique among Bert, Lea, Elisa, Kiki (if present in the dict)
    clique = [n for n in ["Bert", "Lea", "Elisa", "Kiki"] if n in name_to_idx]
    for ai in range(len(clique)):
        for bi in range(ai + 1, len(clique)):
            i = name_to_idx[clique[ai]]
            j = name_to_idx[clique[bi]]
            edges.append([i, j])

    # Only Bert knows Steve (if both exist)
    if "Bert" in name_to_idx and "Steve" in name_to_idx:
        i = name_to_idx["Bert"]
        j = name_to_idx["Steve"]
        if i < j:
            edges.append([i, j])
        elif i > j:
            edges.append([j, i])

    # Convert to edge_index (2, E). Empty if there are no edges.
    edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.empty((2, 0), dtype=torch.long)

    # Return graph
    data = Data(x=x, edge_index=edge_index)
    return data


#%% Define person names and their interests.
# The features are now explicitly defined (e.g., [likes_reading, likes_running, likes_cooking]).
person_interests = {
    "Bert": [0.8, 0.2, 0.2],
    "Lea": [0.1, 0.9, 0.8],
    "Elisa": [0.0, 0.5, 0.8],
    "Kiki": [0.0, 1.0, 0.0],
    "Steve": [0.2, 0.3, 0.5],
}

#%% Generate the dataset
matchmaking_data = create_matchmaking_dataset(person_interests=person_interests)


#%% Print the dataset properties to show what was created
print(f"Number of nodes (people): {matchmaking_data.num_nodes}")
print(f"Number of node features (interests): {matchmaking_data.num_node_features}")
print(f"Number of existing friendships (edges): {matchmaking_data.num_edges}")

print("\n--- Main components of the data object ---")
print(f"data.x: shape {matchmaking_data.x.shape} - The feature matrix for each person.")
print(f"data.edge_index: shape {matchmaking_data.edge_index.shape}")
person_names = list(person_interests.keys())


#%% Simple visualization of the existing friendships graph
def visualize_graph(data: Data, node_names=None):
    """
    Visualize the graph using NetworkX and Matplotlib, including node interests.

    Args:
        data (Data): PyG Data object with at least `edge_index` and node features.
        node_names (List[str], optional): Names for nodes by index.
    """
    G = to_networkx(data, to_undirected=True)
    
    # Create subplots - one for graph, one for interests
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graph visualization
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color="#2C2C54", node_size=2000, edgecolors="#C05C37", ax=ax1)
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax1)

    
    if node_names is not None and len(node_names) == data.num_nodes:
        labels = {i: node_names[i] for i in range(data.num_nodes)}
    else:
        labels = {i: str(i) for i in range(data.num_nodes)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=14, font_color="#FFFFFF", ax=ax1)
    
    ax1.set_title("Friendship Graph")
    ax1.axis("off")

    # Interests visualization
    interests = ["Lesen", "Laufen", "Kochen"]
    x = range(len(node_names))
    width = 0.25
    
    for i, interest in enumerate(interests):
        interests_values = data.x[:, i].numpy()
        ax2.bar([xi + i*width for xi in x], interests_values, width, 
                label=interest, alpha=0.7)
    
    ax2.set_ylabel("Feature Value [-]")
    ax2.set_title("Features of the People")
    ax2.set_xticks([xi + width for xi in x])
    ax2.set_xticklabels(node_names, rotation=45)
    ax2.legend()
    ax2.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.show()

visualize_graph(matchmaking_data, person_names)



# %%
