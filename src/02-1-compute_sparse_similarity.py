import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

hdf5_file_path = "D:\\temp_similarity_data.h5"  # Adjust as needed for testing

def compute_fixed_sparse_similarity_to_hdf5(features, threshold=0.1, batch_size=1000, max_edges=500000):
    """
    Computes cosine similarity in batches, storing only values above threshold in sparse format to HDF5.
    Preallocates a fixed maximum space in the HDF5 file with chunking enabled.
    """
    print("Starting sparse similarity computation...")
    
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        # Preallocate chunked datasets with fixed maximum size to allow resizing
        rows = hdf5_file.create_dataset('row_indices', shape=(max_edges,), maxshape=(None,), dtype='int64', chunks=True)
        cols = hdf5_file.create_dataset('col_indices', shape=(max_edges,), maxshape=(None,), dtype='int64', chunks=True)
        values = hdf5_file.create_dataset('similarity_values', shape=(max_edges,), maxshape=(None,), dtype='float32', chunks=True)
        
        edge_count = 0  # Counter for the number of stored edges

        # Process each batch
        for i in range(0, features.shape[0], batch_size):
            print(f"Processing batch {i // batch_size + 1}...")

            # Compute similarities and print statistics for debugging
            batch_features = features[i:i + batch_size]
            similarity_batch = cosine_similarity(batch_features, features)
            print(f"Batch {i // batch_size + 1} - Min similarity: {similarity_batch.min()}, Max similarity: {similarity_batch.max()}, Mean similarity: {similarity_batch.mean()}")

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
                print("Reached max edges. Stopping...")
                break
            else:
                rows[edge_count:edge_count + len(row_indices)] = row_indices
                cols[edge_count:edge_count + len(col_indices)] = col_indices
                values[edge_count:edge_count + len(similarity_values)] = similarity_values
                edge_count += len(row_indices)
                print(f"Added {len(row_indices)} edges, total edges: {edge_count}")

        # Resize datasets to trim any unused preallocated space
        rows.resize(edge_count, axis=0)
        cols.resize(edge_count, axis=0)
        values.resize(edge_count, axis=0)
        
    print(f"Completed sparse similarity computation with {edge_count} edges.")

# Test the function with a small dataset
if __name__ == "__main__":
    # Example small dataset for testing (replace with real GLCM features)
    test_features = np.random.rand(100, 10)  # Small test sample with 100 nodes and 10 features
    
    compute_fixed_sparse_similarity_to_hdf5(test_features, threshold=0.3, batch_size=20, max_edges=10000)

