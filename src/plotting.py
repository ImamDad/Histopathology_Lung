import matplotlib.pyplot as plt
import numpy as np

# Metrics values based on the model output
precision_per_class = [1.0, 0.66666667, 0.66666667]
recall_per_class = [0.54545455, 0.8, 1.0]
f1_per_class = [0.70588235, 0.72727273, 0.8]
classes = ['Class 0', 'Class 1', 'Class 2']

# Macro averages
macro_metrics = {
    'Precision': 0.7777777777777777,
    'Recall': 0.7818181818181819,
    'F1 Score': 0.744385026737968,
    'ROC AUC': 0.8316985645933014
}

# Plot per-class metrics
plt.figure(figsize=(14, 6))

# Plot Precision, Recall, and F1 Score per class
plt.subplot(1, 2, 1)
x = range(len(classes))
plt.bar(x, precision_per_class, width=0.25, label='Precision', align='center')
plt.bar([p + 0.25 for p in x], recall_per_class, width=0.25, label='Recall', align='center')
plt.bar([p + 0.5 for p in x], f1_per_class, width=0.25, label='F1 Score', align='center')
plt.xticks([p + 0.25 for p in x], classes)
plt.xlabel("Cancer Type")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1 Score per Cancer Type")
plt.legend()

# Plot Macro Averages for Precision, Recall, F1 Score, and ROC AUC
plt.subplot(1, 2, 2)
plt.bar(macro_metrics.keys(), macro_metrics.values(), color=['skyblue', 'orange', 'lightgreen', 'salmon'])
plt.ylim(0, 1)
plt.xlabel("Metric")
plt.ylabel("Score")
plt.title("Macro Averages for Precision, Recall, F1 Score, and ROC AUC")

plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Accuracy values for each subtype
classes = ['ACA', 'SCC', 'BNT']
accuracy_values = [90, 89, 89]

# Plotting the accuracy bar chart
plt.figure(figsize=(12, 6))

# Subplot 1: Bar chart for accuracy
plt.subplot(1, 2, 1)
plt.bar(classes, accuracy_values, color=['blue', 'orange', 'green'])
plt.ylim(0, 100)  # Set the y-axis range to 0-100 for percentage scale
plt.xlabel("Lung Cancer Subtype")
plt.ylabel("Accuracy (%)")
plt.title("Classification Accuracy for Lung Cancer Subtypes")

# Assuming you have 3 classes: 0, 1, 2 for ACA, SCC, BNT
num_classes = 3

# Example data (replace with actual data)
y_true = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2, 1])  # example true labels
y_pred_proba = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.1, 0.85],
                         [0.85, 0.1, 0.05], [0.2, 0.7, 0.1], [0.1, 0.05, 0.85],
                         [0.1, 0.75, 0.15], [0.9, 0.05, 0.05], [0.05, 0.1, 0.85],
                         [0.1, 0.85, 0.05]])

# Binarize the labels for multi-class ROC AUC
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

# Calculate ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Subplot 2: ROC curve for each class
plt.subplot(1, 2, 2)
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# Plot the diagonal line for random chance
plt.plot([0, 1], [0, 1], 'k--', label="Random chance")

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Lung Cancer Subtype Classification')
plt.legend(loc="lower right")

# Show the combined plot
plt.tight_layout()
plt.show()








import matplotlib.pyplot as plt
import numpy as np

# Dummy data for epochs, training accuracy, and validation accuracy
epochs = np.arange(1, 101)
train_accuracy = np.linspace(0.6, 0.9, 100) + np.random.normal(0, 0.01, 100)  # Gradual increase to 90%
val_accuracy = np.linspace(0.55, 0.9, 100) + np.random.normal(0, 0.015, 100)   # Gradual increase to 90%, with slight noise

# Plotting the dummy accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy, label="Training Accuracy", color='blue')
plt.plot(epochs, val_accuracy, label="Validation Accuracy", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)  # Accuracy is between 0 and 1
plt.title("Training and Validation Accuracy Over Epochs")
plt.axhline(y=0.9, color='green', linestyle='--', label="90% Accuracy")
plt.legend()
plt.show()





import matplotlib.pyplot as plt
import numpy as np

# Dummy data for epochs and accuracy
epochs = np.arange(1, 101)  # 1 to 100 epochs
accuracy = 100 * (1 - np.exp(-0.1 * epochs))  # Simulating a curve that asymptotically approaches 100%

# Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracy, label="Accuracy", color='orange', marker='o', markersize=5, linestyle='-')
plt.plot(epochs, epochs, label="Epoch", color='blue', marker='o', markersize=5, linestyle='-')

# Customize the plot
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Over Epochs")
plt.ylim(0, 100)  # Set the y-axis to range from 0% to 100%
plt.legend(loc="lower right")
plt.grid(True)

# Display the plot
plt.show()











#comparision between lung and the breast 

import pandas as pd
import matplotlib.pyplot as plt

# Sample results for Lung Cancer and Breast Cancer datasets
# Replace these values with actual metrics after validating your model on the Breast Cancer dataset
comparison_data = {
    "Dataset": ["Lung Cancer", "Breast Cancer"],
    "Training Accuracy": [0.95, 0.92],
    "Validation Accuracy": [0.88, 0.89],
    "Training Loss": [0.15, 0.18],
    "Validation Loss": [0.22, 0.21]
}

# Create DataFrame to display and save results
comparison_df = pd.DataFrame(comparison_data)

# Save the results to a CSV file with a project-appropriate name
comparison_csv_path = "model_performance_comparison.csv"
comparison_df.to_csv(comparison_csv_path, index=False)
print(f"Comparison results saved to {comparison_csv_path}")

# Plotting the training and validation metrics for both datasets
epochs = range(1, 71)

# Placeholder values for plotting; replace with actual results from training logs if available
lung_train_loss = [0.3 * (0.97 ** epoch) for epoch in epochs]
lung_val_loss = [0.35 * (0.97 ** epoch) for epoch in epochs]
lung_train_accuracy = [min(0.95, 0.8 + 0.0025 * epoch) for epoch in epochs]
lung_val_accuracy = [min(0.88, 0.75 + 0.002 * epoch) for epoch in epochs]

breast_train_loss = [0.35 * (0.96 ** epoch) for epoch in epochs]
breast_val_loss = [0.37 * (0.96 ** epoch) for epoch in epochs]
breast_train_accuracy = [min(0.92, 0.78 + 0.0023 * epoch) for epoch in epochs]
breast_val_accuracy = [min(0.89, 0.76 + 0.0022 * epoch) for epoch in epochs]

# Plotting
plt.figure(figsize=(12, 8))

# Loss comparison for Lung and Breast Cancer
plt.subplot(2, 2, 1)
plt.plot(epochs, lung_train_loss, label="Lung Cancer Training Loss")
plt.plot(epochs, lung_val_loss, label="Lung Cancer Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Lung Cancer Loss")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, breast_train_loss, label="Breast Cancer Training Loss")
plt.plot(epochs, breast_val_loss, label="Breast Cancer Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Breast Cancer Loss")
plt.legend()

# Accuracy comparison for Lung and Breast Cancer
plt.subplot(2, 2, 3)
plt.plot(epochs, lung_train_accuracy, label="Lung Cancer Training Accuracy")
plt.plot(epochs, lung_val_accuracy, label="Lung Cancer Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Lung Cancer Accuracy")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, breast_train_accuracy, label="Breast Cancer Training Accuracy")
plt.plot(epochs, breast_val_accuracy, label="Breast Cancer Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Breast Cancer Accuracy")
plt.legend()

# Save the plot to an appropriate file
plot_path = "performance_comparison_plot.png"
plt.tight_layout()
plt.savefig(plot_path)
plt.show()
print(f"Comparison plot saved to {plot_path}")









#plotting patches 
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

# Define the path to the specific image
image_path = "E:\\project_folder\\Lung_cancer\\data\\lung_scc\\lung_scc_0001.jpg"

# Function to split an image into 64x64 patches
def extract_patches(image, patch_size=(64, 64)):
    patches = []
    for i in range(0, image.shape[0], patch_size[0]):
        for j in range(0, image.shape[1], patch_size[1]):
            patch = image[i:i + patch_size[0], j:j + patch_size[1]]
            if patch.shape[:2] == patch_size:  # Only keep 64x64 patches
                patches.append(patch)
    return patches

# Load the image
image = imread(image_path)

# Extract 64x64 patches without converting to grayscale
patches = extract_patches(image, patch_size=(64, 64))

# Display the number of patches
print(f"Extracted {len(patches)} patches from the image.")

# Optional: Save patches as individual images
output_folder = "E:\\project_folder\\Lung_cancer\\data\\lung_aca_patches"
os.makedirs(output_folder, exist_ok=True)

for idx, patch in enumerate(patches):
    patch_filename = os.path.join(output_folder, f"patch_{idx + 1:03}.png")
    imsave(patch_filename, patch)
    print(f"Saved patch to {patch_filename}")

# Plotting a selection of patches
plt.figure(figsize=(8, 8))
for i in range(min(9, len(patches))):  # Show up to 9 patches
    plt.subplot(3, 3, i + 1)
    plt.imshow(patches[i])
    plt.axis('off')
    plt.title(f"Patch {i + 1}")

plt.tight_layout()
plt.show()




import os
import matplotlib.pyplot as plt
from skimage.io import imread

# Directory containing patch images
patches_dir = "E:\\project_folder\\Lung_cancer\\data\\lung_aca_patches"

# Load a selection of patch images (e.g., first 9 patches)
patch_files = sorted([f for f in os.listdir(patches_dir) if f.endswith('.png')])[:9]
patches = [imread(os.path.join(patches_dir, patch)) for patch in patch_files]

# Plot patches in a 3x3 grid
plt.figure(figsize=(8, 8))
for i, patch in enumerate(patches):
    plt.subplot(3, 3, i + 1)
    plt.imshow(patch, cmap='gray')
    plt.axis('off')
    plt.title(f"Patch {i + 1}")

plt.tight_layout()
plt.show()

