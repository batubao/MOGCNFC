from utils import clean_and_prepare_data, align_all_data, prepare_data
from network import GCN, Classifier
from train import train_and_evaluate
import pandas as pd

# Load datasets
modality_1 = pd.read_csv('/content/1_.csv')
modality_2 = pd.read_csv('/content/2_.csv')
modality_3 = pd.read_csv('/content/3_.csv')
labels = pd.read_csv('/content/labels_.csv')

# Clean and align modalities
modality_1 = clean_and_prepare_data(modality_1)
modality_2 = clean_and_prepare_data(modality_2)
modality_3 = clean_and_prepare_data(modality_3)

modalities, labels = align_all_data([modality_1, modality_2, modality_3], labels)
modality_1, modality_2, modality_3 = modalities

# Prepare data
features_1, labels_tensor = prepare_data(modality_1, labels)
features_2, _ = prepare_data(modality_2, labels)
features_3, _ = prepare_data(modality_3, labels)

# Input dimensions for each modality
input_dim_1 = modality_1.shape[1]
input_dim_2 = modality_2.shape[1]
input_dim_3 = modality_3.shape[1]

# Create models
hidden_dim = 64
output_dim = 32
gcn_1 = GCN(input_dim_1, hidden_dim, output_dim)
gcn_2 = GCN(input_dim_2, hidden_dim, output_dim)
gcn_3 = GCN(input_dim_3, hidden_dim, output_dim)
classifier = Classifier(output_dim * 3, len(labels_tensor.unique()))

# Train and evaluate the model
print("\nTraining and evaluating the model across multiple runs...")
metrics = train_and_evaluate(
    gcn_1, gcn_2, gcn_3, classifier, features_1, features_2, features_3, labels_tensor, runs=10, epochs=100, threshold=0.85
)

# Print final metrics
print("\nFinal Metrics (Mean ± Std):")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value}")
