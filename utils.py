import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def clean_and_prepare_data(data):
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.fillna(data.mean())
    return data

def align_all_data(modalities, labels):
    min_length = min([mod.shape[0] for mod in modalities] + [labels.shape[0]])
    modalities = [mod.iloc[:min_length, :] for mod in modalities]
    labels = labels.iloc[:min_length, :]
    return modalities, labels

def prepare_data(data, labels):
    scaler = StandardScaler()
    features = scaler.fit_transform(data)
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(labels.values.flatten(), dtype=torch.long)
    return features, labels

def create_edge_index_from_features(features, threshold=0.85):
    similarity_matrix = np.matmul(features.numpy(), features.numpy().T) / (
        np.linalg.norm(features.numpy(), axis=1)[:, None]
        * np.linalg.norm(features.numpy(), axis=1)[None, :]
    )
    adjacency_matrix = (similarity_matrix >= threshold).astype(int)
    np.fill_diagonal(adjacency_matrix, 0)
    row, col = np.where(adjacency_matrix > 0)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index
