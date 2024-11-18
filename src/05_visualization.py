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




'''
# 05_visualization.py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def visualize_glcm_features(features, labels, class_names):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    colors = ['red', 'green', 'blue']

    plt.figure(figsize=(10, 8))
    
    # Convert labels to integers for indexing the colors list
    labels = labels.astype(int)
    
    for label in np.unique(labels):
        plt.scatter(reduced_features[labels == label, 0],
                    reduced_features[labels == label, 1],
                    color=colors[label],
                    label=class_names[label],
                    alpha=0.6)

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Visualization of GLCM Features for Lung Cancer Subtypes')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load GLCM features for each cancer type
    scc_features = np.load("E:/project_folder/WSI_Graph_Classification/src/glcm_features_SCC.npy")
    aca_features = np.load("E:/project_folder/WSI_Graph_Classification/src/glcm_features_ACA.npy")  # Adjust path
    bnt_features = np.load("E:/project_folder/WSI_Graph_Classification/src/glcm_features_BNT.npy")  # Adjust path

    # Concatenate features and create labels
    features = np.concatenate([scc_features, aca_features, bnt_features], axis=0)
    labels = np.concatenate([np.zeros(len(scc_features)), 
                             np.ones(len(aca_features)), 
                             np.full(len(bnt_features), 2)])
    
    class_names = ['SCC', 'ACA', 'BNT']  # Class labels

    visualize_glcm_features(features, labels, class_names)






# 05_visualization.py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def visualize_glcm_features(features, labels, class_names):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    colors = ['red', 'green', 'blue']

    plt.figure(figsize=(10, 8))
    
    for label in np.unique(labels):
        plt.scatter(reduced_features[labels == label, 0],
                    reduced_features[labels == label, 1],
                    color=colors[label],
                    label=class_names[label],
                    alpha=0.6)

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Visualization of GLCM Features for Lung Cancer Subtypes')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load GLCM features for each cancer type
    scc_features = np.load("E:/project_folder/WSI_Graph_Classification/src/glcm_features_SCC.npy")
    aca_features = np.load("E:/project_folder/WSI_Graph_Classification/src/glcm_features_ACA.npy")  # Adjust path
    bnt_features = np.load("E:/project_folder/WSI_Graph_Classification/src/glcm_features_BNT.npy")  # Adjust path

    # Concatenate features and create labels
    features = np.concatenate([scc_features, aca_features, bnt_features], axis=0)
    labels = np.concatenate([np.zeros(len(scc_features)), 
                             np.ones(len(aca_features)), 
                             np.full(len(bnt_features), 2)])
    
    class_names = ['SCC', 'ACA', 'BNT']  # Class labels

    visualize_glcm_features(features, labels, class_names)




# 05_visualization.py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_classification(embeddings, labels):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    colors = ['red', 'green', 'blue']
    class_names = ['ACA', 'SCC', 'BNT']

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], color=colors[label], label=class_names[label])

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Graph Visualization of Classified Lung Cancer Subtypes')
    plt.legend(class_names)
    plt.show()
'''