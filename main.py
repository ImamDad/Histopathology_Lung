# main.py
import os
import numpy as np
from src import (
    patch_processing,
    similarity_matrix,
    deepwalk_embeddings,
    gcn_model,
    visualization
)
import torch
from torch_geometric.utils import from_networkx

# Define dataset paths
base_path = "E:\\project_folder\\Lung_cancer\\data\\patches"
aca_path = os.path.join(base_path, "lung_aca")
scc_path = os.path.join(base_path, "lung_scc")
bnt_path = os.path.join(base_path, "lung_bnt")

# Load and process images
aca_images = patch_processing.load_images_from_folder(aca_path)
scc_images = patch_processing.load_images_from_folder(scc_path)
bnt_images = patch_processing.load_images_from_folder(bnt_path)

# Combine images and assign labels
images = aca_images + scc_images + bnt_images
labels = np.array([0] * len(aca_images) + [1] * len(scc_images) + [2] * len(bnt_images))

# Extract GLCM features
glcm_features = patch_processing.extract_glcm_features(images)

# Compute similarity matrix and build graph
adjacency_matrix = similarity_matrix.compute_similarity_matrix(glcm_features)
graph = similarity_matrix.build_graph_from_similarity(adjacency_matrix)

# Generate DeepWalk embeddings
embeddings = deepwalk_embeddings.generate_deepwalk_embeddings(graph)

# Prepare data for GCN
data = from_networkx(graph)
data.x = torch.tensor(embeddings, dtype=torch.float)
data.y = torch.tensor(labels, dtype=torch.long)
data.train_mask = torch.tensor([True] * int(0.8 * len(data.x)) + [False] * int(0.2 * len(data.x)))

# Train GCN
model = gcn_model.GCNClassifier(input_dim=128, hidden_dim=64, num_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
gcn_model.train_gcn(model, data, optimizer)

# Evaluate GCN
accuracy = gcn_model.evaluate_gcn(model, data)
print(f'Accuracy: {accuracy:.4f}')

# Visualize results
visualization.visualize_classification(embeddings, labels)
