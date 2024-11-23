import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from 04_gcn_model import GraphSAGEClassifier


def visualize_classified_graph(graph, true_labels, pred_labels):
    """
    Visualizes the graph nodes and highlights the correctly and incorrectly classified nodes.

    Parameters:
        graph (nx.Graph): The input graph.
        true_labels (numpy.ndarray): Array of true labels.
        pred_labels (numpy.ndarray): Array of predicted labels.
    """
    pos = nx.spring_layout(graph)  # positions for all nodes

    plt.figure(figsize=(12, 12))

    for node in graph.nodes:
        if true_labels[node] == pred_labels[node]:  # Correctly classified
            plt.scatter(pos[node][0], pos[node][1], color='green', label='Correctly Classified', s=100)
        else:  # Incorrectly classified
            plt.scatter(pos[node][0], pos[node][1], color='red', label='Incorrectly Classified', s=100)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, alpha=0.5)

    plt.title("Visualization of Correctly Classified Graph Nodes")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Define model parameters to match those used during training
    input_dim = 256  # Replace with the actual input dimension if different
    hidden_dim = 256  # Set hidden_dim to match the trained model
    num_classes = 3   # Set the number of classes used in classification

    # Load the graph
    graph_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\graph_output.gpickle"
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    # Load the trained model
    model_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\trained_model.pth"
    model = GraphSAGEClassifier(input_dim, hidden_dim, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare the data object (assumes similar structure used during training)
    data = from_networkx(graph)
    true_labels = data.y.cpu().numpy()  # Assumes true labels are present in data.y

    with torch.no_grad():
        # Get predictions from the model
        pred_labels = model(data).argmax(dim=1).cpu().numpy()  # Extract predicted labels

    # Call the visualization function to plot classification results
    visualize_classified_graph(graph, true_labels, pred_labels)

