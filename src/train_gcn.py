
import pickle
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
from torch.nn import functional as F
from sklearn.metrics import accuracy_score

# Define the GraphSAGE model
class GraphSAGEClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GraphSAGEClassifier, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# Modified train function to track both loss and accuracy over epochs
def train_gcn(model, data, optimizer, epochs=500, patience=20):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    class_weights = torch.bincount(data.y[data.train_mask]).float().to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        train_loss.backward()
        optimizer.step()

        # Calculate validation loss
        val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # Calculate training and validation accuracy
        train_pred = out[data.train_mask].argmax(dim=1)
        val_pred = out[data.val_mask].argmax(dim=1)
        train_acc = accuracy_score(data.y[data.train_mask].cpu(), train_pred.cpu())
        val_acc = accuracy_score(data.y[data.val_mask].cpu(), val_pred.cpu())
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return train_losses, val_losses, train_accuracies, val_accuracies

# Plotting function to visualize the training and validation loss and accuracy
def plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Sample setup to run the training process and plot the learning curve
if __name__ == "__main__":
    # Define the paths to your graph and embeddings
    graph_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\graph_output.gpickle"
    embedding_output_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\deepwalk_embeddings.npy"

    # Load graph and embeddings
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    data = from_networkx(graph)
    data.x = torch.tensor(np.load(embedding_output_path), dtype=torch.float)

    # Dummy labels and masks for demonstration
    data.y = torch.randint(0, 3, (data.num_nodes,), dtype=torch.long)
    data.train_mask = torch.rand(data.num_nodes) < 0.6
    data.val_mask = (torch.rand(data.num_nodes) >= 0.6) & (torch.rand(data.num_nodes) < 0.8)
    data.test_mask = torch.rand(data.num_nodes) >= 0.8

    # Move data and model to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = data.to(device)

    # Initialize model and optimizer
    input_dim = data.x.shape[1]
    hidden_dim = 256
    num_classes = len(data.y.unique())
    model = GraphSAGEClassifier(input_dim, hidden_dim, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
'''
    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_gcn(model, data, optimizer, epochs=500, patience=50)

    # Plot the learning curve
    plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies)
    '''


# Train the model and capture the metrics
train_losses, val_losses, train_accuracies, val_accuracies = train_gcn(model, data, optimizer, epochs=500, patience=50)

# Combine the data into a DataFrame
results_df = pd.DataFrame({
    'Epoch': range(1, len(train_losses) + 1),
    'Training Loss': train_losses,
    'Validation Loss': val_losses,
    'Training Accuracy': train_accuracies,
    'Validation Accuracy': val_accuracies
})

# Save the DataFrame to a CSV file
results_csv_path = "training_metrics.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Training metrics saved to {results_csv_path}")

# Plotting function to visualize the training and validation loss and accuracy and save the plot as an image
def plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies, save_path="learning_curve.png"):
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    
    # Save the plot as an image file
    plt.savefig(save_path)
    plt.show()
    print(f"Learning curve plot saved as {save_path}")

# Call the plotting function and save the plot as an image
plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies)















