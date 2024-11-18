import networkx as nx
import h5py
import pickle

def build_graph_from_sparse_hdf5(hdf5_file_path, graph_output_path):
    """
    Builds a graph by reading sparse similarity values from the HDF5 file and adding edges.
    """
    print("Starting graph construction from HDF5 data...")
    graph = nx.Graph()

    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        rows = hdf5_file['row_indices'][:]
        cols = hdf5_file['col_indices'][:]
        values = hdf5_file['similarity_values'][:]

        print(f"Number of edges to add: {len(rows)}")

        # Add edges to the graph
        for row, col, similarity in zip(rows, cols, values):
            if row != col:  # Avoid self-loops
                graph.add_edge(row, col, weight=similarity)

    print(f"Graph construction completed. Number of nodes: {graph.number_of_nodes()}, Number of edges: {graph.number_of_edges()}")

    # Save the graph using pickle
    with open(graph_output_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Graph saved to {graph_output_path}")

# Set the paths for HDF5 input and .gpickle output
hdf5_file_path = "D:\\temp_similarity_data.h5"
graph_output_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\graph_output.gpickle"

# Execute the graph construction and save it
build_graph_from_sparse_hdf5(hdf5_file_path, graph_output_path)




'''
import os
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import h5py

# Paths to GLCM feature files for each subtype
feature_files = {
    "ACA": "glcm_features_ACA.npy",
    "SCC": "glcm_features_SCC.npy",
    "BNT": "glcm_features_BNT.npy"
}

# HDF5 file to store sparse similarity information
hdf5_file_path = "D:\\temp_fixed_sparse_similarity_data.h5"  # Adjust to a drive with more space if needed

def compute_fixed_sparse_similarity_to_hdf5(features, threshold=0.5, batch_size=1000, max_edges=500000):
    """
    Computes cosine similarity in batches, storing only values above threshold in sparse format to HDF5.
    Preallocates a fixed maximum space in the HDF5 file.
    """
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        # Preallocate datasets with fixed maximum size to avoid resizing issues
        rows = hdf5_file.create_dataset('row_indices', shape=(max_edges,), dtype='int64')
        cols = hdf5_file.create_dataset('col_indices', shape=(max_edges,), dtype='int64')
        values = hdf5_file.create_dataset('similarity_values', shape=(max_edges,), dtype='float32')
        
        edge_count = 0  # Counter for the number of stored edges

        for i in range(0, features.shape[0], batch_size):
            batch_features = features[i:i + batch_size]
            similarity_batch = cosine_similarity(batch_features, features)

            # Find indices where similarity is above threshold
            row_indices, col_indices = np.where(similarity_batch > threshold)
            similarity_values = similarity_batch[row_indices, col_indices]
            row_indices += i  # Adjust indices for batch offset

            # Store only as many edges as the preallocated space allows
            if edge_count + len(row_indices) > max_edges:
                remaining_space = max_edges - edge_count
                rows[edge_count:edge_count + remaining_space] = row_indices[:remaining_space]
                cols[edge_count:edge_count + remaining_space] = col_indices[:remaining_space]
                values[edge_count:edge_count + remaining_space] = similarity_values[:remaining_space]
                break
            else:
                rows[edge_count:edge_count + len(row_indices)] = row_indices
                cols[edge_count:edge_count + len(col_indices)] = col_indices
                values[edge_count:edge_count + len(similarity_values)] = similarity_values
                edge_count += len(row_indices)

        # Resize datasets to trim any unused preallocated space
        rows.resize(edge_count, axis=0)
        cols.resize(edge_count, axis=0)
        values.resize(edge_count, axis=0)

def build_graph_from_sparse_hdf5(hdf5_file_path):
    """
    Builds a graph by reading sparse similarity values from the HDF5 file and adding edges.
    """
    graph = nx.Graph()
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        rows = hdf5_file['row_indices'][:]
        cols = hdf5_file['col_indices'][:]
        values = hdf5_file['similarity_values'][:]

        # Add edges to the graph based on sparse data
        for row, col, similarity in zip(rows, cols, values):
            if row != col:  # Avoid self-loops
                graph.add_edge(row, col, weight=similarity)

    return graph

if __name__ == "__main__":
    print("Running 02_similarity_matrix.py")

    for label, file_path in feature_files.items():
        if not os.path.exists(file_path):
            print(f"Feature file does not exist for {label}: {file_path}")
            continue

        # Load GLCM features
        print(f"Loading GLCM features for {label}...")
        features = np.load(file_path)

        # Compute and store sparse similarity in HDF5 format
        print(f"Computing sparse similarity with fixed preallocation for {label}...")
        compute_fixed_sparse_similarity_to_hdf5(features, threshold=0.5, batch_size=1000, max_edges=500000)

        # Build the graph from sparse HDF5 data
        print(f"Building graph for {label} from sparse HDF5 data...")
        graph = build_graph_from_sparse_hdf5(hdf5_file_path)

        # Save the constructed graph
        graph_filename = f"graph_{label}.gpickle"
        nx.write_gpickle(graph, graph_filename)
        print(f"Saved graph for {label} to {graph_filename}")

    # Clean up HDF5 file after processing
    os.remove(hdf5_file_path)
    print("Sparse similarity matrix computation and graph construction completed for all subtypes.")
    '''
