

import pickle
import numpy as np

# Define the file paths
graph_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\graph_output.gpickle"
embedding_output_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\deepwalk_embeddings.npy"

# Check dimensions of the files
# Load graph structure from .gpickle file and check the number of nodes and edges
with open(graph_path, "rb") as f:
    graph = pickle.load(f)
print("Graph Information:")
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")

# Load embeddings from .npy file and check its shape
embeddings = np.load(embedding_output_path)
print("Embeddings Shape:", embeddings.shape)