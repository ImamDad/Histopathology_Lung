import networkx as nx
import numpy as np
import torch
import torch_cluster
print(torch_cluster.__version__)
from torch_geometric.utils import from_networkx
from torch_geometric.nn import Node2Vec
import pickle

# Path to the saved graph file
graph_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\graph_output.gpickle"
# Path to save the embeddings
embedding_output_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\deepwalk_embeddings.npy"

def generate_deepwalk_embeddings(graph, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10, epochs=100, lr=0.01):
    """
    Generates DeepWalk embeddings using Node2Vec.
    Parameters:
        - graph: NetworkX graph to process.
        - embedding_dim: Dimension of the embeddings.
        - walk_length: Length of each random walk.
        - context_size: Size of the context window.
        - walks_per_node: Number of walks to start from each node.
        - epochs: Number of epochs to train Node2Vec.
        - lr: Learning rate for the optimizer.
    Returns:
        - embeddings: NumPy array of node embeddings.
    """
    # Convert NetworkX graph to PyTorch Geometric format
    data = from_networkx(graph)
    node2vec = Node2Vec(data.edge_index, embedding_dim=embedding_dim, walk_length=walk_length, context_size=context_size, walks_per_node=walks_per_node, sparse=True)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    node2vec = node2vec.to(device)
    optimizer = torch.optim.SparseAdam(node2vec.parameters(), lr=lr)

    # Train the Node2Vec model
    def train_node2vec():
        node2vec.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Define a batch of nodes on the CPU, then move to the appropriate device
            batch = torch.arange(data.num_nodes, device='cpu')
            pos_rw, neg_rw = node2vec.sample(batch)
            
            # Move random walks to the correct device
            pos_rw, neg_rw = pos_rw.to(device), neg_rw.to(device)
            
            # Calculate loss and perform backpropagation
            loss = node2vec.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    train_node2vec()

    # Extract embeddings and convert to NumPy array
    embeddings = node2vec.embedding.weight.detach().cpu().numpy()
    return embeddings

if __name__ == "__main__":
    print("Loading graph for embedding generation...")
    
    # Load the existing graph using pickle
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    # Generate embeddings
    embeddings = generate_deepwalk_embeddings(graph, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10, epochs=100, lr=0.01)

    # Save embeddings to file
    np.save(embedding_output_path, embeddings)
    print(f"DeepWalk embeddings saved to {embedding_output_path}")




# plotting

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Load the embeddings
file_path = "E:/project_folder/WSI_Graph_Classification/src/deepwalk_embeddings.npy"
deepwalk_embeddings = np.load(file_path)

# Step 1: Dimensionality Reduction with PCA and t-SNE

# Apply PCA to reduce to 50 dimensions initially (for faster t-SNE)
pca = PCA(n_components=50)
pca_embeddings = pca.fit_transform(deepwalk_embeddings)

# Apply t-SNE to project to 2D space
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_embeddings = tsne.fit_transform(pca_embeddings)

# Step 2: Clustering with KMeans
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters as needed
clusters = kmeans.fit_predict(deepwalk_embeddings)

# Plotting the t-SNE projection with KMeans clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Cluster Label")
plt.title("t-SNE Visualization of DeepWalk Embeddings with KMeans Clusters")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()


