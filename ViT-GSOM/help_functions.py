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
import matplotlib.gridspec as gridspec


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
    y_coords, x_coords = torch.meshgrid( torch.arange(row_num, dtype=torch.float32),
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

            grid_dists = torch.norm(bmu1_coords - bmu2_coords, p=2, dim=1)
            total_te += torch.sum(grid_dists > 1.42).item()

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
    output["Purity"] = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    return output



def get_node_hits(model: 'AutoEncoder',
                           loader: torch.utils.data.DataLoader,
                           device: torch.device) -> np.ndarray:
    """
    Function which calculates the number of times each neuron on the grid becomes a BMU
    :param model: ViT-SOM Autoencoder
    :param loader: Dataloader
    :param device: The torch device
    :return: Numpy array with the number of hits for each neuron of shape ``(n_nodes,number of unique labels)``
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


    return node_hits

def plot_umap_som_weights(snapshot_som_weights, unique_labels: set):
    """
    Function which plots the SOM weights for different epochs using UMAP visualization
    :param snapshot_som_weights: Dictionary containing snapshot of SOM weights for different epochs
    :param unique_labels: Set with all unique labels
    """
    color_options = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:pink',
                     'tab:olive', 'tab:cyan', 'tab:gray']
    cmap = colors.ListedColormap(color_options)
    cmap.set_bad(color='black')
        
    for epoch, (weights, node_hits) in snapshot_som_weights.items():
        # get label with max votes
        node_labels = np.argmax(node_hits, axis=1)

        # units with no votes
        total_hits = np.sum(node_hits, axis=1)
        node_labels[total_hits == 0] = -1

        reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric='cosine', random_state=42, n_jobs=1)
        embedding = reducer.fit_transform(weights)
        active_mask = node_labels != -1

        plt.figure(figsize=(10, 8))
        # print empty nodes as black
        if np.sum(~active_mask) > 0:
            plt.scatter(embedding[~active_mask, 0], embedding[~active_mask, 1], c='black')

        # plot active nodes
        if np.sum(active_mask) > 0:
            scatter = plt.scatter(embedding[active_mask, 0], embedding[active_mask, 1],
                                  c=node_labels[active_mask], cmap=cmap)

        # create patches for the legend
        patches = [mpatches.Patch(color=color_options[i], label=f"Class {i}") for i in range(len(unique_labels))]
        patches.append(mpatches.Patch(color='black', label='Empty'))
        
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"SOM UMAP weights with majority class, epoch {epoch}")
        plt.show()
        
def plot_som_weights(snapshot_som_weights,
                     som_rows: int,
                     som_cols: int,
                     unique_labels: set):
    """
    Function which plots the SOM weights for different epochs
    :param snapshot_som_weights: Dictionary containing snapshot of SOM weights for different epochs
    :param som_rows: Number of rows on the grid
    :param som_cols: Number of columns on the grid,
    :param unique_labels: Set with all unique labels
    """

    # define colors
    color_options = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:pink',
                     'tab:olive', 'tab:cyan', 'tab:gray']
    cmap = colors.ListedColormap(color_options)
    cmap.set_bad(color='black')

    for epoch, (weights, node_hits) in snapshot_som_weights.items():
        # get label with max votes
        node_labels = np.argmax(node_hits, axis=1)

        # units with no votes
        total_hits = np.sum(node_hits, axis=1)
        node_labels[total_hits == 0] = -1
        
        matrix = node_labels.reshape(som_rows, som_cols)
        masked_matrix = np.ma.masked_where(matrix == -1, matrix)
        
        # create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(masked_matrix, cmap=cmap, vmin=0, vmax=9)
        ax.set_title(f"SOM nodes with majority class, epoch: {epoch}")

        # create patches for the legend
        patches = [mpatches.Patch(color=color_options[i], label=f"Class {i}") for i in range(len(unique_labels))]
        patches.append(mpatches.Patch(color='black', label='Empty'))
    
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

def plot_som_pie_grid(snapshot_som_weights, 
                      som_rows: int, 
                      som_cols: int, 
                      unique_labels: int):
    """
    Function which plots the SOM grid with its class distributions
    :param snapshot_som_weights: Dictionary containing snapshot of SOM weights for different epochs
    :param som_rows: Number of rows on the grid
    :param som_cols: Number of columns on the grid,
    :param unique_labels: Set with all unique labels
    """
    
    color_options = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:pink',
                     'tab:olive', 'tab:cyan', 'tab:gray']
    
    fig = plt.figure(figsize=(10, 10))
    the_grid = gridspec.GridSpec(som_rows, som_cols, fig)

    node_hits = snapshot_som_weights[list(snapshot_som_weights.keys())[-1]][1]

    for i in range(som_rows * som_cols):   
        row = i // som_cols
        col = i % som_cols     
        counts = node_hits[i]
        
        ax = plt.subplot(the_grid[row, col])

        if np.sum(counts) == 0:
            ax.pie([1], colors=['black'])
        else:
            ax.pie(counts, colors=color_options[:len(unique_labels)])

    patches = [mpatches.Patch(color=color_options[i], label=f"Class {i}") for i in range(len(unique_labels))]
    patches.append(mpatches.Patch(color='black', label='Empty'))
    
    fig.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle(f"SOM nodes with class distributions ({som_rows}x{som_cols})", fontsize=16)
    plt.show()

def plot_som_mnist(snapshot_som_weights: np.ndarray, som_rows: int, som_cols: int):
    """
    Function for MNIST dataset which plots the SOM grid with the majority label as a text
    :param snapshot_som_weights: Dictionary containing snapshot of SOM weights for different epochs
    :param som_rows: Number of rows on the grid
    :param som_cols: Number of columns on the grid,
    """

    color_options = ['tab:green', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:pink',
                     'tab:olive', 'tab:cyan', 'tab:gray']
    
    fig, ax = plt.subplots(figsize=(8, 8))

    node_hits = snapshot_som_weights[list(snapshot_som_weights.keys())[-1]][1]

    node_labels = np.argmax(node_hits, axis=1)

    # units with no votes
    total_hits = np.sum(node_hits, axis=1)
    node_labels[total_hits == 0] = -1

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax.set_xlim(0, som_cols)
    ax.set_ylim(0, som_rows)
    ax.invert_yaxis()
    
    for i in range(som_rows * som_cols):
        label = node_labels[i]
        
        if label != -1:
            row = i // som_cols
            col = i % som_cols
            
            center_x = col + 0.5
            center_y = row + 0.5
            c = color_options[int(label) % len(color_options)]
            
            ax.text(center_x, center_y, str(label), color=c, fontdict={'weight': 'bold', 'size': 12})
    
    plt.title(f"SOM nodes with majority class ({som_rows}x{som_cols})")
    plt.show()

def generate_extended_u_matrix(weight_matrix: np.ndarray, rows, cols) -> np.ndarray:
    """
    Function to generate Extended U-Matrix
    :param weight_matrix: Weights matrix from the model
    :return: Extended U-Matrix
    """
    weight_matrix = weight_matrix.reshape((rows, cols, -1))
    m, n, dim = weight_matrix.shape

    # Size of extended U-Matrix (2m-1)*(2n-1)
    ext_m = 2 * m - 1
    ext_n = 2 * n - 1
    u_matrix_extended = np.zeros((ext_m, ext_n))

    # Horizontal distances
    for r in range(m):
        for c in range(n - 1):
            current_node = weight_matrix[r, c]
            right_neighbour = weight_matrix[r, c + 1]

            dist = np.linalg.norm(current_node - right_neighbour)
            u_matrix_extended[2 * r, 2 * c + 1] = dist

    # Vertical distances
    for r in range(m - 1):
        for c in range(n):
            current_node = weight_matrix[r, c]
            bottom_neighbour = weight_matrix[r + 1, c]

            dist = np.linalg.norm(current_node - bottom_neighbour)
            u_matrix_extended[2 * r + 1, 2 * c] = dist

    # Centers among distances - odd rows, odd columns - average over neighbours
    for r in range(1, ext_m, 2):
        for c in range(1, ext_n, 2):
            neighbours = [u_matrix_extended[r - 1, c], u_matrix_extended[r + 1, c], u_matrix_extended[r, c - 1],
                         u_matrix_extended[r, c + 1]]
            u_matrix_extended[r, c] = np.mean(neighbours)

    # Neuron positions - average over neighbours
    for r in range(0, ext_m, 2):
        for c in range(0, ext_n, 2):
            distances = []
            if r > 0: distances.append(u_matrix_extended[r - 1, c]) #top
            if r < ext_m - 1: distances.append(u_matrix_extended[r + 1, c]) #bottom
            if c > 0: distances.append(u_matrix_extended[r, c - 1]) #left
            if c < ext_n - 1: distances.append(u_matrix_extended[r, c + 1]) #right

            u_matrix_extended[r,c] = np.mean(distances)

    return u_matrix_extended

def visualize_u_matrix_extended(matrix: np.ndarray, text : str = ""):
    """
    Function to visualize U-Matrix
    :param matrix: Numpy Extended U-Matrix matrix
    :param text: Text for the title
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(matrix, cmap='plasma')

    m, n = matrix.shape
    ax.set_xticks(np.arange(0, n, 2))
    ax.set_yticks(np.arange(0, m, 2))
    ax.set_xticklabels(np.arange(len(np.arange(0, n, 2))))
    ax.set_yticklabels(np.arange(len(np.arange(0, m, 2))))

    neuron_x_coords = []
    neuron_y_coords = []

    for r in range(0, m, 2):
        for c in range(0, n, 2):
            neuron_y_coords.append(r)
            neuron_x_coords.append(c)

    ax.scatter(neuron_x_coords, neuron_y_coords, s=30, c='yellow', edgecolors='black', label='Neuron Position')

    ax.set_title(f"U-Matrix Extended, {text}")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Distance')
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))

    plt.tight_layout()
    plt.show()