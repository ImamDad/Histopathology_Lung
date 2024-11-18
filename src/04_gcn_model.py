'''
# 04_gcn_model.py

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

def train_gcn(model, data, optimizer, epochs=300, patience=20):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    class_weights = torch.bincount(data.y[data.train_mask]).float().to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        train_loss.backward()
        optimizer.step()

        val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        print(f'Epoch {epoch}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return train_losses, val_losses
torch.save(model.state_dict(), "E:\\project_folder\\WSI_Graph_Classification\\src\\trained_model.pth")

def evaluate_gcn(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        accuracy = int(correct) / int(data.test_mask.sum())
        return accuracy, pred[data.test_mask], data.y[data.test_mask]

def plot_metrics(train_losses, val_losses, test_accuracy, y_true, y_pred):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_true.cpu(), y_pred.cpu())
    ConfusionMatrixDisplay(cm).plot(values_format='d', cmap="Blues", ax=plt.gca())
    plt.title(f"Confusion Matrix (Test Accuracy: {test_accuracy:.2f})")
    plt.show()

def calculate_metrics(y_true, y_pred, num_classes=3):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true.cpu(), y_pred.cpu())
    metrics['precision'] = precision_score(y_true.cpu(), y_pred.cpu(), average=None)
    metrics['recall'] = recall_score(y_true.cpu(), y_pred.cpu(), average=None)
    metrics['f1'] = f1_score(y_true.cpu(), y_pred.cpu(), average=None)
    metrics['precision_macro'] = precision_score(y_true.cpu(), y_pred.cpu(), average='macro')
    metrics['recall_macro'] = recall_score(y_true.cpu(), y_pred.cpu(), average='macro')
    metrics['f1_macro'] = f1_score(y_true.cpu(), y_pred.cpu(), average='macro')

    try:
        y_true_bin = np.eye(num_classes)[y_true.cpu()]
        y_pred_bin = np.eye(num_classes)[y_pred.cpu()]
        metrics['roc_auc'] = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovr')
    except Exception as e:
        metrics['roc_auc'] = None

    return metrics

if __name__ == "__main__":
    graph_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\graph_output.gpickle"
    embedding_output_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\deepwalk_embeddings.npy"

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    data = from_networkx(graph)
    data.x = torch.tensor(np.load(embedding_output_path), dtype=torch.float)
    
    data.y = torch.randint(0, 3, (data.num_nodes,), dtype=torch.long)
    data.train_mask = torch.rand(data.num_nodes) < 0.6
    data.val_mask = (torch.rand(data.num_nodes) >= 0.6) & (torch.rand(data.num_nodes) < 0.8)
    data.test_mask = torch.rand(data.num_nodes) >= 0.8

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = data.to(device)

    input_dim = data.x.shape[1]
    hidden_dim = 256
    num_classes = len(data.y.unique())
    model = GraphSAGEClassifier(input_dim, hidden_dim, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    print("Training the GraphSAGE model with early stopping...")
    train_losses, val_losses = train_gcn(model, data, optimizer, epochs=300, patience=20)

    test_accuracy, y_pred, y_true = evaluate_gcn(model, data)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    plot_metrics(train_losses, val_losses, test_accuracy, y_true, y_pred)

    metrics = calculate_metrics(y_true, y_pred, num_classes=num_classes)
    print("Detailed metrics for each cancer type:")
    print("Accuracy:", metrics['accuracy'])
    print("Precision per class:", metrics['precision'])
    print("Recall per class:", metrics['recall'])
    print("F1 score per class:", metrics['f1'])
    print("Macro Precision:", metrics['precision_macro'])
    print("Macro Recall:", metrics['recall_macro'])
    print("Macro F1 Score:", metrics['f1_macro'])
    print("ROC AUC (Macro):", metrics['roc_auc'])



'''
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

def train_gcn(model, data, optimizer, epochs=300, patience=20):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    class_weights = torch.bincount(data.y[data.train_mask]).float().to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        train_loss.backward()
        optimizer.step()

        val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        print(f'Epoch {epoch}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return train_losses, val_losses

def evaluate_gcn(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        accuracy = int(correct) / int(data.test_mask.sum())
        return accuracy, pred[data.test_mask], data.y[data.test_mask]

def plot_metrics(train_losses, val_losses, test_accuracy, y_true, y_pred):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_true.cpu(), y_pred.cpu())
    ConfusionMatrixDisplay(cm).plot(values_format='d', cmap="Blues", ax=plt.gca())
    plt.title(f"Confusion Matrix (Test Accuracy: {test_accuracy:.2f})")
    plt.show()

def calculate_metrics(y_true, y_pred, num_classes=3):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true.cpu(), y_pred.cpu())
    metrics['precision'] = precision_score(y_true.cpu(), y_pred.cpu(), average=None)
    metrics['recall'] = recall_score(y_true.cpu(), y_pred.cpu(), average=None)
    metrics['f1'] = f1_score(y_true.cpu(), y_pred.cpu(), average=None)
    metrics['precision_macro'] = precision_score(y_true.cpu(), y_pred.cpu(), average='macro')
    metrics['recall_macro'] = recall_score(y_true.cpu(), y_pred.cpu(), average='macro')
    metrics['f1_macro'] = f1_score(y_true.cpu(), y_pred.cpu(), average='macro')

    try:
        y_true_bin = np.eye(num_classes)[y_true.cpu()]
        y_pred_bin = np.eye(num_classes)[y_pred.cpu()]
        metrics['roc_auc'] = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovr')
    except Exception as e:
        metrics['roc_auc'] = None

    return metrics

if __name__ == "__main__":
    graph_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\graph_output.gpickle"
    embedding_output_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\deepwalk_embeddings.npy"

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    data = from_networkx(graph)
    data.x = torch.tensor(np.load(embedding_output_path), dtype=torch.float)
    
    data.y = torch.randint(0, 3, (data.num_nodes,), dtype=torch.long)
    data.train_mask = torch.rand(data.num_nodes) < 0.6
    data.val_mask = (torch.rand(data.num_nodes) >= 0.6) & (torch.rand(data.num_nodes) < 0.8)
    data.test_mask = torch.rand(data.num_nodes) >= 0.8

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = data.to(device)

    input_dim = data.x.shape[1]
    hidden_dim = 256
    num_classes = len(data.y.unique())
    model = GraphSAGEClassifier(input_dim, hidden_dim, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    print("Training the GraphSAGE model with early stopping...")
    train_losses, val_losses = train_gcn(model, data, optimizer, epochs=300, patience=20)

    test_accuracy, y_pred, y_true = evaluate_gcn(model, data)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    plot_metrics(train_losses, val_losses, test_accuracy, y_true, y_pred)

    # Save the model after training
    torch.save(model.state_dict(), "E:\\project_folder\\WSI_Graph_Classification\\src\\trained_model.pth")  # Save model state




import pickle
import numpy as np

# Define the file paths
graph_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\graph_output.gpickle"
embedding_output_path = "E:\\project_folder\\WSI_Graph_Classification\\src\\deepwalk_embeddings.npy"

# Check dimensions of the files
# Load graph structure from .gpickle file and check the number of nodes and edges
with open(graph_path, "rb") as f:
    graph = pickle.load(f)
print("Graph Information:")
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")

# Load embeddings from .npy file and check its shape
embeddings = np.load(embedding_output_path)
print("Embeddings Shape:", embeddings.shape)




