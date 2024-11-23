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


