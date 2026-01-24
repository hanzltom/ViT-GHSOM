import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn import metrics
import umap
import matplotlib.pyplot as plt
import copy

"""
Distance functions
"""

def euclidean_distance(a, b, axis):
    diff = np.abs(a - b)
    return np.linalg.norm(diff, axis=axis)

def manhattan_distance(a, b, axis):
    diff = np.abs(a - b)
    return np.sum(diff, axis=axis)

def chebyshev_distance(a, b, axis):
    diff = np.abs(a - b)
    return np.max(diff, axis=axis)

def generic_distance(a, b, axis, k):
    diff = np.abs(a - b)
    return np.power(np.sum(np.power(diff, k), axis=axis), 1.0 / k)

def cosine_distance_torch(weights, inputs):
    # eg. 3x3 grid with weights of dim 4: (3,3,4) -> (9,4)
    if weights.ndim == 3:
        weights_flat = weights.reshape(-1, weights.shape[-1])
    else:
        weights_flat = weights

    # input size: (batch size, dim size), e.g. (32,4)
    inputs_norm = F.normalize(inputs, dim=1)
    # (9,4)
    weights_norm = F.normalize(weights_flat, dim=1)

    # e.g. (32,4) dot (4,9) = (32,9)
    similarity = torch.mm(inputs_norm, weights_norm.t())

    return 1 - similarity

def cosine_distance_numpy(weights, inputs):
    # eg. 3x3 grid with weights of dim 4: (3,3,4) -> (9,4)
    if weights.ndim == 3:
        weights_flat = weights.reshape(-1, weights.shape[-1])
    else:
        weights_flat = weights

    # eg input (4,) -> (1,4)
    if inputs.ndim == 1:
        inputs = inputs[np.newaxis, :]

    # input dot weights: (1,4) dot (4,9) = (1,9)
    divident = np.dot(inputs, weights_flat.T)

    input_norm = np.linalg.norm(inputs,axis=1, keepdims=True) # (1,1)
    weight_norm = np.linalg.norm(weights_flat,axis=1, keepdims=True) # (9,1)
    divisor = input_norm * weight_norm.T # (1,1) * (1,9) = (1,9)


    similarity = divident / (divisor + 1e-8)
    return 1 - similarity


"""
Neighbourhood distance functions
"""
def gaussian_neighbourhood_numpy(grid_dists, sigma_t):
    return np.exp(- (grid_dists ** 2) / (2 * (sigma_t ** 2)))

def gaussian_neighbourhood_torch(grid_dists, sigma_t):
    return torch.exp(-grid_dists / (2 * sigma_t ** 2))

def rectangular_neighbourhood(grid_dists, sigma_t):
    return (grid_dists <= sigma_t).astype(float)

def triangular_neighbourhood(grid_dists, sigma_t):
    return np.maximum(0.0, 1.0 - (grid_dists / sigma_t))

def cosine_down_to_zero_neighbourhood(grid_dists, sigma_t):
    influence = np.zeros_like(grid_dists)
    mask = grid_dists <= 2 * sigma_t

    influence[mask] = (np.cos((np.pi * grid_dists[mask]) / (2 * sigma_t)) + 1) / 2.0
    return influence


"""
Decay function
"""

def decay_exponential(initial_value, beta, t):
    return initial_value * (beta ** t)


def decay_power(initial_value, beta, t):
    return initial_value * (t ** beta)

"""
Other functions
"""
def get_grid_coords(row_num, col_num, device):
    y_coords, x_coords = torch.meshgrid(
torch.arange(row_num, dtype=torch.float32),
        torch.arange(col_num, dtype=torch.float32),
        indexing='ij'
    )

    # coords are 2 dim tensors, we stack them over new dimension to shape (nrows, ncol, 2)
    # reshape them to shape (num_units, 2)
    coords = torch.stack((x_coords, y_coords), dim=-1).reshape(-1, 2)
    return coords.to(device)

def calculate_purity(model, loader, device):
# https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
    model.eval()
    true_label = []
    cluster_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            _, latent = model(images)

            # extract cls token with sequence of patches - not needed
            # shape (batch, embed_dim)
            #latent = latent[:,0,:]
            patches = latent[:, 1:, :] 
            
            # 2. Flatten: (Batch, 784)
            som_input = patches.reshape(patches.shape[0], -1)

            # calculate distance, shape (batch, neuron unit num)
            dists = cosine_distance_torch(model.get_som_weights(), som_input)

            bmu_indices = torch.argmin(dists, dim=1)
            true_label.append(labels)
            cluster_labels.append(bmu_indices)

    true_labels_np = torch.cat(true_label).cpu().numpy()
    cluster_labels_np = torch.cat(cluster_labels).cpu().numpy()
    
    contingency_matrix = metrics.cluster.contingency_matrix(true_labels_np, cluster_labels_np)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def capture_latent(model, loader, device):
    model.eval()
    labels_vector = []
    latent_vectors = []
    cls_vectors = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            _, latent = model(images)
            cls_token = latent[:, 0, :]
            latent = latent[:,1:,:]
            
            # (Batch, 49, 16) -> (Batch, 784)
            latent = latent.reshape(latent.shape[0], -1)
            
            cls_vectors.append(cls_token.cpu().numpy())
            latent_vectors.append(latent.cpu().numpy())
            labels_vector.append(labels.cpu().numpy())

    X_patches = np.concatenate(latent_vectors, axis=0)
    X_cls = np.concatenate(cls_vectors, axis=0)
    y = np.concatenate(labels_vector, axis=0)

    model.train()

    return X_patches, X_cls, y


def plot_umap_patches(snapshot):
    for epoch, (patches,y) in snapshot.items():

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(patches)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='tab10')
        plt.colorbar(scatter, ticks=range(10), label='Digit Class')
        plt.title(f"UMAP patches, epoch: {epoch}")
        plt.show()

def plot_umap_cls(snapshot):
    for epoch, (cls, y) in snapshot.items():

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(cls)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='tab10')
        plt.colorbar(scatter, ticks=range(10), label='Digit Class')
        plt.title(f"UMAP CLS token, epoch: {epoch}")
        plt.show()


def get_node_labels(model, loader, device):
    rows, cols = model.get_som_shape()
    num_nodes = rows * cols

    node_hits = np.zeros((num_nodes, 10))

    for images, labels in loader:
        images = images.to(device)
        labels = labels.cpu().numpy()

        _, latent = model(images)
        patches = latent[:, 1:, :]
        som_input = patches.reshape(patches.shape[0], -1)

        # find bmu for batch
        dists = cosine_distance_torch(model.get_som_weights(), som_input)
        bmu_indices = torch.argmin(dists, dim=1).cpu().numpy()

        # add vote to neuron
        for i, bmu_idx in enumerate(bmu_indices):
            node_hits[bmu_idx, labels[i]] += 1

    # get label with max votes
    node_labels = np.argmax(node_hits, axis=1)

    # units with no votes
    total_hits = np.sum(node_hits, axis=1)
    node_labels[total_hits == 0] = -1

    return node_labels

def plot_umap_som_weights(snapshot_som_weights):
    for epoch, (weights, labels) in snapshot_som_weights.items():
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(weights)
        active_mask = labels != -1

        plt.figure(figsize=(10, 8))
        if np.sum(active_mask) > 0:
            scatter = plt.scatter(embedding[active_mask, 0], embedding[active_mask, 1],
                                  c=labels[active_mask], cmap='tab10')
            plt.colorbar(scatter, ticks=range(10), label='Digit Class')

        if np.sum(~active_mask) > 0:
            plt.scatter(embedding[~active_mask, 0], embedding[~active_mask, 1], c='black')

        plt.title(f"SOM weights, epoch {epoch}")
        plt.show()

