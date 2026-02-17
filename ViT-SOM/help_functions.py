import math

import numpy
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn import metrics
import umap
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

"""
Distance functions
"""

def cosine_distance_torch(weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Function calculating cosine distance between weights and inputs
    :param weights: SOM weights of shape ``(som_rows * som_cols, num_patches * embed_dim)``
    :param inputs: Input batch tensor of shape ``(batch_size, n_features)``
    :return: Distance of shape ``(batch_size, som_rows * som_cols)``
    """
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


"""
Neighbourhood distance functions
"""

def gaussian_neighbourhood_torch(grid_dists: torch.Tensor, sigma_t: float) -> torch.Tensor:
    """
    Function calculating gaussian neighbourhood influence
    :param grid_dists: Squared Euclidean distances between the BMU and other neurons of shape ``(batch_size, n_nodes)``
    :param sigma_t: Current neighbourhood radius
    :return: Neighbourhood influence tensor of shape ``(batch_size, n_nodes)``
    """
    return torch.exp(-grid_dists / (2 * sigma_t ** 2))


"""
Decay function
"""

def decay_exponential(initial_value: float, beta: float, t: int) -> float:
    """
    Decay exponential function
    :param initial_value: Initial value
    :param beta: Beta value, must satisfy: 0 < beta < 1
    :param t: Current time
    :return: Decayed initial value
    """
    return initial_value * (beta ** t)


def decay_power(initial_value: float, beta: float, t: int) -> float:
    """
    Decay power function
    :param initial_value: Initial value
    :param beta: Beta value, must satisfy: beta < 0
    :param t: Current time
    :return: Decayed initial value
    """
    return initial_value * (t ** beta)

"""
Other functions
"""
def get_grid_coords(row_num: int, col_num: int, device: torch.device | str) -> torch.Tensor:
    """
    Function calculating grid of 2D coordinates for the SOM
    :param row_num: Number of rows in the SOM grid
    :param col_num: Number of columns in the SOM grid
    :param device: The torch device
    :return: Grid coordinates of shape ``(row_num * col_num, 2)``.
    """
    y_coords, x_coords = torch.meshgrid(
torch.arange(row_num, dtype=torch.float32),
        torch.arange(col_num, dtype=torch.float32),
        indexing='ij'
    )

    # coords are 2 dim tensors, we stack them over new dimension to shape (row_num, col_num, 2)
    # reshape them to shape (num_units, 2)
    coords = torch.stack((x_coords, y_coords), dim=-1).reshape(-1, 2)
    return coords.to(device)

def calculate_QE_TE_Purity(model: 'AutoEncoder',
                           loader: torch.utils.data.DataLoader,
                           device: torch.device) -> dict[str: float]:
    """
    Function calculating QE, TE and Purity metrics for model evaluation
    :param model: Trained ViT-SOM Autoencoder
    :param loader: Dataloader
    :param device: The torch device
    :return: A tuple containing (QE, TE, Purity)
    """
# https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
    model.eval()
    true_label = []
    cluster_labels = []
    total_qe = 0.0
    total_te = 0.0
    total_samples = 0

    rows, cols = model.get_som_shape()
    grid_coords = get_grid_coords(rows, cols, device)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            _, latent = model(images)

            # extract cls token with sequence of patches - not needed
            # shape (batch, embed_dim)
            patches = latent[:, 1:, :]
            
            # flatten to create som input
            som_input = patches.reshape(patches.shape[0], -1)

            # calculate distance, shape (batch, neuron unit num)
            dists = cosine_distance_torch(model.get_som_weights(), som_input)
            min_dists, bmu_indices = torch.min(dists, dim=1)

            # QE
            total_qe += torch.sum(min_dists).item()

            # TE
            _, top2_indices = torch.topk(dists, k=2, dim=1, largest=False)
            bmu1_idx = top2_indices[:, 0]
            bmu2_idx = top2_indices[:, 1]

            bmu1_coords = grid_coords[bmu1_idx]
            bmu2_coords = grid_coords[bmu2_idx]

            grid_dists = torch.cdist(bmu1_coords, bmu2_coords)
            total_te += torch.sum(grid_dists > np.sqrt(2)).item()

            # Purity
            true_label.append(labels.cpu())
            cluster_labels.append(bmu_indices.cpu())

            total_samples += dists.shape[0]

    output = {}
    output["QE"] = total_qe / total_samples if total_samples > 0 else 0
    output["TE"] = total_te / total_samples if total_samples > 0 else 0

    true_labels_np = torch.cat(true_label).cpu().numpy()
    cluster_labels_np = torch.cat(cluster_labels).cpu().numpy()
    contingency_matrix = metrics.cluster.contingency_matrix(true_labels_np, cluster_labels_np)
    output["Purity"] =  np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    return output


def capture_latent(model: 'AutoEncoder',
                           loader: torch.utils.data.DataLoader,
                           device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Functions which captures current latent embedding for visualization
    :param model: ViT-SOM Autoencoder
    :param loader: Dataloader
    :param device: The torch device
    :return: A tuple containing:
             - X_patches: Flattened patch embeddings of shape ``(n_samples, num_patches * embed_dim)``
             - X_cls: CLS token embeddings of shape ``(n_samples, embed_dim)``
             - y: Target labels of shape ``(n_samples,)``
    """
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


def plot_umap_patches(snapshot: dict[int: tuple[torch.Tensor, torch.Tensor]]):
    """
    Functions which plots the patches for different epochs using UMAP visualization
    :param snapshot: Dictionary containing snapshot of patches for different epochs
    """
    for epoch, (patches,y) in snapshot.items():

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(patches)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='tab10')
        plt.colorbar(scatter, ticks=range(10), label='Digit Class')
        plt.title(f"UMAP patches, epoch: {epoch}")
        plt.show()

def plot_umap_cls(snapshot: dict[int: tuple[torch.Tensor, torch.Tensor]]):
    """
    Functions which plots the CLS tokens for different epochs using UMAP visualization
    :param snapshot: Dictionary containing snapshot of CLS tokens for different epochs
    """
    for epoch, (cls, y) in snapshot.items():

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(cls)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='tab10')
        plt.colorbar(scatter, ticks=range(10), label='Digit Class')
        plt.title(f"UMAP CLS token, epoch: {epoch}")
        plt.show()


def get_node_labels(model: 'AutoEncoder',
                           loader: torch.utils.data.DataLoader,
                           device: torch.device) -> np.ndarray:
    """
    Function which calculates the majority class for each neuron on the grid
    :param model: ViT-SOM Autoencoder
    :param loader: Dataloader
    :param device: The torch device
    :return: Numpy array with the label of the majority class of shape ``(n_nodes,)``
    """
    model.eval()
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
        np.add.at(node_hits, (bmu_indices, labels), 1)

    # get label with max votes
    node_labels = np.argmax(node_hits, axis=1)

    # units with no votes
    total_hits = np.sum(node_hits, axis=1)
    node_labels[total_hits == 0] = -1

    return node_labels

def plot_umap_som_weights(snapshot_som_weights: dict[int: tuple[torch.Tensor, numpy.ndarray]]):
    """
    Functions which plots the SOM weights for different epochs using UMAP visualization
    :param snapshot_som_weights: Dictionary containing snapshot of CLS tokens for different epochs
    """
    for epoch, (weights, labels) in snapshot_som_weights.items():
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(weights)
        active_mask = labels != -1

        plt.figure(figsize=(10, 8))
        # plot active nodes
        if np.sum(active_mask) > 0:
            scatter = plt.scatter(embedding[active_mask, 0], embedding[active_mask, 1],
                                  c=labels[active_mask], cmap='tab10')
            plt.colorbar(scatter, ticks=range(10), label='Digit Class')

        # print empty nodes as black
        if np.sum(~active_mask) > 0:
            plt.scatter(embedding[~active_mask, 0], embedding[~active_mask, 1], c='black')

        plt.title(f"SOM weights, epoch {epoch}")
        plt.show()

def plot_som_weights(snapshot_som_weights,
                     som_rows: int,
                     som_cols: int,
                     unique_labels: set):
    """
    Functions which plots the SOM weights for different epochs
    :param snapshot_som_weights: Dictionary containing snapshot of CLS tokens for different epochs
    :param som_rows: Number of rows on the grid
    :param som_cols: Number of columns on the grid,
    :param unique_labels: Set with all unique labels
    """

    # define colors
    color_options = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:pink',
                     'tab:olive', 'tab:cyan', 'tab:gray']
    cmap = colors.ListedColormap(color_options)
    cmap.set_bad(color='black')

    for epoch, (weights, labels) in snapshot_som_weights.items():
        matrix = labels.reshape(som_rows, som_cols)
        masked_matrix = np.ma.masked_where(matrix == -1, matrix)
        
        # create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(masked_matrix, cmap=cmap, vmin=0, vmax=9)
        ax.set_title(f"Epoch: {epoch}")

        # create patches for the legend
        patches = []
        for label in unique_labels:
            if label == -1:
                # add empty black node
                patches.append(mpatches.Patch(color='black', label='Empty'))
            else:
                color_idx = int(label) % len(color_options)
                patches.append(mpatches.Patch(color=color_options[color_idx], label=f"Class {label}"))
        
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
