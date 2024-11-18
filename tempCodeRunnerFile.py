import matplotlib.pyplot as plt
import numpy as np

# Sample results from the previous code
models = ['CNN', 'GraphSAGE', 'GAT', 'Proposed GCN']
accuracy = [cnn_accuracy, sage_accuracy, gat_accuracy, gcn_accuracy]
f1_scores = [cnn_f1, sage_f1, gat_f1, gcn_f1]
inference_times = [cnn_inference_time, sage_inference_time, gat_inference_time, gcn_inference_time]

# Set up figure for subplots
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# Plot Accuracy
ax[0].bar(models, accuracy, color=['blue', 'green', 'purple', 'orange'])
ax[0].set_title('Model Accuracy')
ax[0].set_xlabel('Model')
ax[0].set_ylabel('Accuracy')
ax[0].set_ylim(0, 1)  # Assuming accuracy is between 0 and 1

# Plot F1 Score
ax[1].bar(models, f1_scores, color=['blue', 'green', 'purple', 'orange'])
ax[1].set_title('Model F1 Score')
ax[1].set_xlabel('Model')
ax[1].set_ylabel('F1 Score')
ax[1].set_ylim(0, 1)  # Assuming F1 score is between 0 and 1

# Plot Inference Time
ax[2].bar(models, inference_times, color=['blue', 'green', 'purple', 'orange'])
ax[2].set_title('Model Inference Time')
ax[2].set_xlabel('Model')
ax[2].set_ylabel('Inference Time (s/sample)')
ax[2].set_ylim(0, max(inference_times) * 1.1)  # Set a bit higher than max inference time

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
