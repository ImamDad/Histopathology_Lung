import importlib.util
import torch
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import numpy as np
from torch_geometric.utils import from_networkx
import torch
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv

# Dynamically load the GraphSAGEClassifier
spec = importlib.util.spec_from_file_location("GraphSAGEClassifier", "E:\\project_folder\\WSI_Graph_Classification\\src\\04_gcn_model.py")
gcn_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gcn_model)
GraphSAGEClassifier = gcn_model.GraphSAGEClassifier

# Load the trained model
model_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\trained_model.pth"
model = GraphSAGEClassifier(input_dim=128, hidden_dim=256, num_classes=3)  # Adjust to match trained model
model.load_state_dict(torch.load(model_path))
model.eval()

# Load the graph
graph_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\graph_output.gpickle"
with open(graph_path, "rb") as f:
    graph = pickle.load(f)

# Convert the graph to PyTorch Geometric format
data = from_networkx(graph)

# Load embeddings and labels (assumed to be available)
embeddings = torch.tensor(np.load("E:\\project_folder\\WSI_Graph_Classification\\src\\deepwalk_embeddings.npy"), dtype=torch.float)
data.x = embeddings

# Simulating labels for illustration; replace with actual labels
data.y = torch.randint(0, 3, (data.num_nodes,), dtype=torch.long)  # Replace this line with actual labels if available

# Make predictions
with torch.no_grad():
    out = model(data)
    pred_labels = out.argmax(dim=1)

# Visualizing the classified nodes
def visualize_classified_graph(graph, true_labels, pred_labels):
    plt.figure(figsize=(12, 12))
    
    pos = nx.spring_layout(graph)  # positions for all nodes
    
    # Draw nodes based on true labels
    nx.draw_networkx_nodes(graph, pos, node_color=true_labels, cmap=plt.cm.RdYlBu, alpha=0.5)
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    
    # Draw labels for nodes
    labels = {i: str(i) for i in range(len(true_labels))}
    nx.draw_networkx_labels(graph, pos, labels, font_size=12)
    
    # Draw correctly classified nodes
    for node in range(len(true_labels)):
        if true_labels[node] == pred_labels[node]:
            plt.scatter(pos[node][0], pos[node][1], s=200, edgecolor='black', facecolor='lime', alpha=0.7)

    plt.title('Graph Visualization of Correctly Classified Nodes')
    plt.axis('off')  # Turn off the axis
    plt.show()

# Call the visualization function
true_labels = data.y.cpu().numpy()  # Assuming labels are in 'data.y'
pred_labels = pred_labels.cpu().numpy()  # Move predictions to CPU for visualization
visualize_classified_graph(graph, true_labels, pred_labels)










