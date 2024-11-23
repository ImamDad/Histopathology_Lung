Here's the updated `README.md` with the execution flow steps added:

---

# WSI Graph Classification

This project proposes a graph-based framework for classifying lung cancer subtypes—adenocarcinoma (ACA), squamous cell carcinoma (SCC), and benign tissue (BNT)—in Whole Slide Images (WSIs). Combining texture analysis, graph embeddings, and Graph Convolutional Networks (GCNs), this approach models spatial dependencies between tissue regions, significantly improving accuracy and interpretability in digital pathology.

## Project Structure

```plaintext
WSI_Graph_Classification/
├── data/
│   ├── aca/                         # Folder containing ACA patches
│   ├── scc/                         # Folder containing SCC patches
│   └── bnt/                         # Folder containing BNT patches
├── src/
│   ├── 01_patch_processing.py       # Script for GLCM feature extraction
│   ├── 02_similarity_matrix.py      # Script for cosine similarity and graph construction
│   ├── 02-1-compute_sparse_similarity.py # Computes sparse similarity matrix
│   ├── 03_deepwalk_embeddings.py    # Script for DeepWalk embedding generation
│   ├── train_gcn.py                # Defines the proposed GCN model
│   ├── 05_visualization.py          # Script for visualizing graph structure and results
├── compare_sota_models.py           # Main script to train and evaluate SOTA models
├── plotting.py                      # Script to plot model comparison results
├── requirements.txt                 # List of dependencies
└── README.md                        # Project overview and instructions
```

## Setup Instructions

1. **Clone or Download the Project**: Clone this repository or download the project folder.

   ```bash
   git clone <your-repo-url>
   cd WSI_Graph_Classification
   ```

2. **Set Up a Virtual Environment** (optional):

   ```bash
   python -m venv env
   source env/bin/activate   # On Windows, use: env\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

1. **Organize Dataset**: Ensure your WSI patches are organized in separate folders under `data/` for each subtype (ACA, SCC, BNT).
   
   ```plaintext
   data/
   ├── aca/         # Adenocarcinoma patches
   ├── scc/         # Squamous cell carcinoma patches
   └── bnt/         # Benign tissue patches
   ```

## Execution Flow

Follow these steps to run the files in the correct sequence:

1. **Feature Extraction**:
   - Run `01_patch_processing.py` to extract GLCM features.
   - **Command**: `python src/01_patch_processing.py`

2. **Sparse Similarity Matrix Computation**:
   - Run `02-1-compute_sparse_similarity.py` to compute the sparse cosine similarity matrix.
   - **Command**: `python src/02-1-compute_sparse_similarity.py`

3. **Graph Construction**:
   - Execute `02_similarity_matrix.py` to build the weighted graph from similarity values.
   - **Command**: `python src/02_similarity_matrix.py`

4. **Embedding Generation**:
   - Run `03_deepwalk_embeddings.py` to generate node embeddings using DeepWalk.
   - **Command**: `python src/03_deepwalk_embeddings.py`

5. **Train and Evaluate the GCN Model**:
   - Run `train_gcn.py` to train and evaluate the proposed GraphSAGE model.
   - **Command**: `python src/04_gcn_model.py`

6. **Compare with State-of-the-Art Models**:
   - Execute `compare_sota_models.py` to evaluate the proposed model alongside CNN, GraphSAGE, and GAT models.
   - **Command**: `python compare_sota_models.py`

7. **Visualization**:
   - Use `05_visualization.py` to visualize the graph structure and classification results.
   - **Command**: `python src/05_visualization.py`

8. **Model Comparison Plotting**:
   - Run `plotting.py` to generate comparative bar plots for accuracy, F1 score, and inference time.
   - **Command**: `python plotting.py`

9. **Run Main Script**:
   - Alternatively, use `main.py` to sequentially run all steps.
   - **Command**: `python main.py`

## Key Files and Functions

- **`01_patch_processing.py`**: Extracts texture features (GLCM) from each patch.
- **`02_similarity_matrix.py`**: Computes cosine similarity and constructs the weighted adjacency matrix.
- **`03_deepwalk_embeddings.py`**: Generates DeepWalk embeddings to encode graph structure.
- **`train_gcn.py`**: Defines the architecture of the proposed GCN model.
- **`compare_sota_models.py`**: Trains and evaluates CNN, GraphSAGE, GAT, and the proposed GCN model.
- **`05_visualization.py`**: Visualizes graph structure with classification labels.

## Results

The framework outputs accuracy, F1 scores, and inference times across models, with visualizations indicating the classification accuracy for ACA, SCC, and BNT. The proposed GCN model should demonstrate competitive or superior performance compared to SOTA models, illustrating the benefits of graph-based classification for WSI analysis.

## Requirements

Key libraries:
- `torch`
- `torch_geometric`
- `scikit-learn`
- `matplotlib`
- `numpy`

See `requirements.txt` for specific versions.

---

This README now includes all necessary steps to set up, run, and evaluate the project along with the correct file execution sequence. Let me know if you need further customization!
