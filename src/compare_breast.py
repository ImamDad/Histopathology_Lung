# File: E:\project_folder\WSI_Graph_Classification\src\compare_breast.py

import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define paths for benign and malignant folders
benign_folder = r"F:\8 GB data\archive (5)\Multi Cancer\Breast Cancer\breast_benign"
malignant_folder = r"F:\8 GB data\archive (5)\Multi Cancer\Breast Cancer\breast_malignant"

# Define a transform for data preprocessing and augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),               # Resize images to a standard size
    transforms.RandomHorizontalFlip(),           # Apply random horizontal flip
    transforms.RandomRotation(10),               # Apply random rotation within 10 degrees
    transforms.ToTensor(),                       # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale image (adjust if needed)
])

# Custom Dataset class to load images and labels from directories
class BreastCancerDataset(Dataset):
    def __init__(self, benign_dir, malignant_dir, transform=None):
        self.benign_dir = benign_dir
        self.malignant_dir = malignant_dir
        self.transform = transform
        
        # List all .jpg images in benign and malignant folders
        self.benign_images = [os.path.join(benign_dir, img) for img in os.listdir(benign_dir) if img.endswith('.jpg')]
        self.malignant_images = [os.path.join(malignant_dir, img) for img in os.listdir(malignant_dir) if img.endswith('.jpg')]
        
        # Combine lists and create labels: 0 for benign, 1 for malignant
        self.all_images = self.benign_images + self.malignant_images
        self.labels = [0] * len(self.benign_images) + [1] * len(self.malignant_images)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.labels[idx]
        
        # Load image and apply transforms
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Create Dataset and DataLoader instances for training, validation, and test sets
dataset = BreastCancerDataset(benign_folder, malignant_folder, transform=data_transforms)

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Data loading complete. Dataloaders created for training, validation, and test sets.")

# Enhanced GraphSAGE model for classification
class EnhancedGraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(EnhancedGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Generate synthetic node features and edges for demonstration
node_features = torch.tensor(np.random.rand(len(dataset), 128), dtype=torch.float)  # Replace with actual features
edge_index = torch.tensor([[i, j] for i in range(len(dataset)) for j in range(i + 1, len(dataset)) if i != j][:100], dtype=torch.long).t()

# Create the graph data object
data = Data(x=node_features, edge_index=edge_index.contiguous(), y=torch.tensor(dataset.labels))

# Splitting into train, validation, and test masks
train_mask = torch.rand(len(data.y)) < 0.7
val_mask = (torch.rand(len(data.y)) >= 0.7) & (torch.rand(len(data.y)) < 0.85)
test_mask = torch.rand(len(data.y)) >= 0.85

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

train_loader = GraphDataLoader([data], batch_size=1, shuffle=True)
val_loader = GraphDataLoader([data], batch_size=1, shuffle=False)
test_loader = GraphDataLoader([data], batch_size=1, shuffle=False)

# Initialize the model, loss function, and optimizer
input_dim = node_features.shape[1]
hidden_dim = 128
num_classes = 2
model = EnhancedGraphSAGE(input_dim, hidden_dim, num_classes)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train():
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

# Validation function
def validate():
    model.eval()
    correct = 0
    for data in val_loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred[data.val_mask] == data.y[data.val_mask]).sum()
    val_accuracy = int(correct) / int(data.val_mask.sum())
    print(f'Validation Accuracy: {val_accuracy:.4f}')

# Train and validate the model over epochs
num_epochs = 10
for epoch in range(num_epochs):
    train()
    validate()

# Testing function
def test():
    model.eval()
    correct = 0
    preds, labels = [], []
    for data in test_loader:
