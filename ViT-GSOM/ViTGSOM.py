import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from help_functions import cosine_distance_torch, gaussian_neighbourhood_torch


"""
https://tintn.github.io/Implementing-Vision-Transformer-from-Scratch/
https://www.geeksforgeeks.org/deep-learning/how-to-use-pytorchs-nnmultiheadattention/
https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/
"""

class SomLoss(nn.Module):
    """
    Class to calculate SOM loss for weights update
    """
    def __init__(self):
        """
        Constructor for SomLoss class
        """
        super().__init__()

    def forward(self,
                latent_vectors: torch.Tensor,
                som_weights: torch.Tensor,
                grid_coords: torch.Tensor,
                sigma: float) -> torch.Tensor:
        """
        Forward pass to compute SOM loss
        :param latent_vectors: Patch embeddings from the encoder of shape ``(batch_size, seq_len + 1, embed_dim)``
        :param som_weights: SOM weight tensor of shape ``(n_nodes, n_features)``
        :param grid_coords: Grid coordinate tensor of shape ``(n_nodes, 2)``
        :param sigma: The current neighborhood radius
        :return: Scalar loss tensor
        """
        # latent vector shape: (batch, sequence of patches + cls, embed_dim), cls not needed for SOM, only patches
        patches = latent_vectors[:, 1:, :]

        som_input = patches.reshape(patches.shape[0], -1)

        # distance for all samples in batch, shape (batch, Num_Units)
        dists = cosine_distance_torch(som_weights, som_input)

        # indices of bmu for each sample in batch, size (batch,)
        bmu_indices = torch.argmin(dists, dim=1)

        # coordinates of the bmus for this batch, shape (batch, 2)
        bmu_coords = grid_coords[bmu_indices]

        # calculating euclidean distance between bmus and all other neuron units along the coordinate dimension
        # unsqueezing to allow broadcasting
        # (batch, 1, 2) - (1, Num_Units, 2) -> (batch, Num_Units, 2)
        dist_grid = torch.sum((bmu_coords.unsqueeze(1) - grid_coords.unsqueeze(0)) ** 2, dim=2)

        # calculating neighbourhood influence through neighbourhood function - gaussian
        neighbourhood_influence = gaussian_neighbourhood_torch(dist_grid, sigma)

        loss = neighbourhood_influence * dists
        return loss.sum(dim=1).mean() # Equation 3

class ViTLoss(nn.Module):
    """
    Class to calculate ViT loss for weights update
    """
    def __init__(self):
        """
        Constructor for ViTLoss class
        """
        super().__init__()
        
        self.mseLoss = nn.MSELoss()

    def forward(self, original_img: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute ViT loss using MSE
        :param original_img: The real input image of shape ``(batch_size, channels, height, width)``
        :param reconstructed: The reconstructed image by the decoder of shape ``(batch_size, channels, height, width)``
        :return: Scalar loss tensor
        """
        l_nn = self.mseLoss(original_img, reconstructed)

        return l_nn


def unpatch(x: torch.Tensor, patch_size: int = 4, channels: int = 1) -> torch.Tensor:
    """
    Function which transforms the patches from the decoder back to its original input size
    :param x: Patch embeddings of shape ``(batch_size, num_patches, embed_dim)``
    :param patch_size: Size of the patch. Defaults to ``4``
    :param channels: Number of input channels. Defaults to ``1``
    :return: Sequence of picture in original input size
    """    # E.g. (8, 49, 3*4*4): batch of 8, 7x7 grid, num_of_channels * patch_size * patch_size
    B, num_patches, pixels_per_patch = x.shape

    if pixels_per_patch != channels * patch_size * patch_size:
        raise ValueError(f'Number of pixels in patch {pixels_per_patch} must be equal to channels * patch_size * patch_size: {channels * patch_size * patch_size}')

    # get size of the grid
    # sqrt(49) = 7 -> 7x7 grid of patches
    grid_h = int(num_patches ** 0.5)
    grid_w = int(num_patches ** 0.5)

    # (B, 49, 48) -> (B, 49, 3, 4, 4): (batch, num_patches, num_of_channels, patch_height, patch_width)
    x = x.reshape(B, num_patches, channels, patch_size, patch_size)

    # (B, 49, 3, 4, 4) -> (B, 7, 7, 3, 4, 4): (Batch, grid_H, grid_W, num_of_channels, patch_H, patch_W)
    x = x.reshape(B, grid_h, grid_w, channels, patch_size, patch_size)

    # (Batch, grid_H, grid_W, num_of_channels, patch_H, patch_W) -> (Batch, num_of_channels, grid_H, patch_H, grid_W, patch_W)
    # (B, 7, 7, 3, 4, 4) -> (B, 3, 7, 4, 7, 4)
    x = x.permute(0, 3, 1, 4, 2, 5)

    # get original size of image
    # (B, 3, 7, 4, 7, 4) -> (B, 3, 7 * 4, 7 * 4)
    x = x.reshape(B, channels, grid_h * patch_size, grid_w * patch_size)

    return x

class PatchEmbedding(nn.Module):
    """
    Class to split images into patches and embed them using convolutional layers
    """
    def __init__(self,
                 img_size: int = 28,
                 patch_size: int = 4,
                 in_channels: int = 1,
                 embed_dim: int = 16):
        """
        Constructor for PatchEmbedding
        :param img_size: Size of input image. Defaults to ``28``
        :param patch_size: Size of individual patch. Defaults to ``4``
        :param in_channels: Number of input channels. Defaults to ``1``
        :param embed_dim: The embedding dimension. Defaults to ``16``
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # convolution with the stride size same as patch size -> no overlapping
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to create patch embeddings
        :param x: Input batch of images of shape ``(batch_size, in_channels, img_size, img_size)``
        :return: Patch embeddings of shape ``(batch_size, num_patches, embed_dim)``
        """
        # Example: batch = 8, embed_dim=64, img_height=28, img_width=28, input_channels=1, patch_size=4
        # x.shape: (8, 1, 28, 28)
        # 28 / 4 = 7 -> 7x7 grid
        # proj(x).shape: (8, 64, 7, 7)
        # proj(x).flatten(2): (8, 64, 7 * 7)
        # proj(x).flatten(2).transpose(1, 2): (8, 49, 16) : (B, 7x7 grid as sequence, embed_dim)
        x = self.proj(x).flatten(2)
        x = x.transpose(1, 2)
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron class
    """
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float):
        """
        Constructor for MLP
        :param embed_dim: The embedding dimension
        :param mlp_dim: The dimension of the hidden layer
        :param dropout: The dropout probability
        """
        super().__init__()
        self.dense_1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP
        :param x: Input tensor of shape ``(batch_size, seq_len, embed_dim)``
        :return: Output tensor of shape ``(batch_size, seq_len, embed_dim)``
        """
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    Transformer Encoder Block consisting of Self-Attention and MLP
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 dropout: float = 0.1):
        """
        Constructor for Block
        :param embed_dim: The embedding dimension
        :param num_heads: The number of attention heads
        :param mlp_dim: The dimension of the hidden layer
        :param dropout: The dropout probability. Defaults to ``0.1``
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer Encoder Block
        :param x: Input tensor of shape ``(batch_size, seq_len, embed_dim)``
        :return: Output tensor of shape ``(batch_size, seq_len, embed_dim)``
        """
        # Self-attention
        attention_output, _ = self.attention(self.ln1(x), self.ln1(x), self.ln1(x))
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.ln2(x))
        # Skip connection
        x = x + mlp_output
        return x

class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder.
    """

    def __init__(self,
                 img_size: int = 28,
                 patch_size: int = 4,
                 in_channels: int = 1,
                 embed_dim: int = 16,
                 depth: int = 4,
                 num_heads: int = 2,
                 mlp_dim: int = 64):
        """
        Constructor for ViTEncoder
        :param img_size: Size of input image. Defaults to ``28``
        :param patch_size: Size of individual patch. Defaults to ``4``
        :param in_channels: Number of input channels. Defaults to ``1``
        :param embed_dim: The embedding dimension. Defaults to ``16``
        :param depth: The number of transformer encoder blocks. Defaults to ``4``
        :param num_heads: The number of attention heads. Defaults to ``2``
        :param mlp_dim: The dimension of the hidden layer. Defaults to ``64``
        """
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # learnable positional embedding and cls token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            block = Block(embed_dim, num_heads, mlp_dim)
            self.blocks.append(block)

        self.ln1 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ViTEncoder
        :param x: Input images of shape ``(batch_size, in_channels, img_size, img_size)``
        :return: Encoded features of shape ``(batch_size, num_patches + 1, embed_dim)``
        """
        # create patches
        # x shape: (batch_size, num_patches, embed_dim)
        B = x.shape[0]
        x = self.patch_embed(x)

        # Add CLS token
        # cls_tokens shape: (batch_size, 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # x shape: (batch_size, num_patches + 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add Positional Embedding
        x = x + self.pos_embed

        # apply self-attention layers and mlp
        for block in self.blocks:
            x = block(x)

        x = self.ln1(x)
        return x



class ViTDecoder(nn.Module):
    """
    Vision Transformer Decoder.
    """

    def __init__(self,
                 num_patches: int,
                 patch_size: int = 4,
                 output_dim: int = 1,
                 embed_dim: int = 16,
                 depth: int = 2,
                 num_heads: int = 2,
                 mlp_dim: int = 64):
        """
        Constructor for ViTDecoder
        :param num_patches: Total number of patches
        :param patch_size: Size of individual patch. Defaults to ``4``
        :param output_dim: Number of output channels. Defaults to ``1``
        :param embed_dim: The embedding dimension. Defaults to ``16``
        :param depth: The number of transformer encoder blocks. Defaults to ``4``
        :param num_heads: The number of attention heads. Defaults to ``2``
        :param mlp_dim: The dimension of the hidden layer. Defaults to ``64``
        """
        super().__init__()

        # reconstruction to original pixels: patch_size * patch_size * channels
        self.pixels_per_patch = patch_size * patch_size * output_dim
        self.num_patches = num_patches
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # same list of transformer blocks as in encoder
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            block = Block(embed_dim, num_heads, mlp_dim)
            self.blocks.append(block)

        self.ln1 = nn.LayerNorm(embed_dim)

        # final projection to map embeddings back to pixel values
        self.head = nn.Linear(embed_dim, self.pixels_per_patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ViTDecoder
        :param x: Encoded features of shape ``(batch_size, num_patches + 1, embed_dim)``
        :return: Reconstructed patches of shape ``(batch_size, num_patches, pixels_per_patch)``
        """
        # positional embeddings in latent space
        x = x + self.pos_embed

        # applying self attention layers and mlp
        for block in self.blocks:
            x = block(x)

        x = self.ln1(x)
        # removing CLS token (8, 50, 64) -> (8, 49, 64)
        x = x[:, 1:, :]

        # projection back to pixel space (8, 49, 64) -> (8, 49, 16): (B, grid 7x7, pixels per patch)
        x = self.head(x)
        return x

class AutoEncoder(nn.Module):
    """
    Vision Autoencoder with an integrated Self-Organizing Map layer
    """
    def __init__(self,
                 img_size: int = 28,
                 patch_size: int = 4,
                 num_of_channels: int = 1,
                 embed_dim: int = 16,
                 enc_depth: int = 4,
                 dec_depth: int = 2,
                 num_heads: int = 2,
                 mlp_dim: int = 64,
                 som_rows: int = 5,
                 som_cols: int = 5):
        """
        Constructor for AutoEncoder
        :param img_size: Size of input image. Defaults to ``28``
        :param patch_size: Size of individual patch. Defaults to ``4``
        :param num_of_channels: Number of input channels. Defaults to ``1``
        :param embed_dim: The embedding dimension. Defaults to ``16``
        :param enc_depth: The number of transformer encoder blocks. Defaults to ``4``
        :param dec_depth: The number of transformer decoder blocks. Defaults to ``2``
        :param num_heads: The number of attention heads. Defaults to ``2``
        :param mlp_dim: The dimension of the hidden layer. Defaults to ``64``
        :param som_rows: Number of rows in the SOM grid
        :param som_cols: Number of columns in the SOM grid
        """
        super().__init__()

        # ensure image size can be divided by the size of the patch
        if img_size % patch_size != 0:
            raise ValueError(f"Image size ({img_size}) must be divisible by patch size ({patch_size}).")

        self.num_of_channels = num_of_channels
        self.patch_size = patch_size
        self.num_of_patches = (img_size // patch_size) ** 2

        # Encoder: Image -> Latent
        self.encoder = ViTEncoder(img_size, patch_size, num_of_channels, embed_dim, enc_depth, num_heads, mlp_dim)
        # Decoder: Latent -> Reconstructed patches
        self.decoder = ViTDecoder(self.num_of_patches, patch_size, num_of_channels, embed_dim, dec_depth, num_heads, mlp_dim)

        self.current_row_num = som_rows
        self.current_col_num = som_cols
        # SOM weights as a torch Parameter
        # shape (Num_SOM_Nodes, Num_Patches * Embed_Dim)
        self.som_dim = self.num_of_patches * embed_dim
        self.som_weights = nn.Parameter(torch.randn(self.current_row_num * self.current_col_num, self.som_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for AutoEncoder
        :param x: Input images of shape ``(batch_size, in_channels, img_size, img_size)``
        :return: Tuple containing:
                 - output: Reconstructed images of shape ``(batch_size, num_channels, img_size, img_size)``
                 - latent: Latent representations of shape ``(batch_size, num_patches + 1, embed_dim)``
        """
        latent = self.encoder(x)
        patched_output = self.decoder(latent)
        output = unpatch(patched_output, self.patch_size, self.num_of_channels)
        return output, latent

    def get_sigma(self) -> float:
        """
        Calculates the initial sigma for the SOM as half of the image size
        :return: Sigma value
        """
        return np.ceil(min(self.som_rows, self.som_cols) / 2)

    def get_som_shape(self) -> tuple[int, int]:
        """
        Returns the shape of the SOM grid
        :return: A tuple (rows, cols)
        """
        return self.som_rows, self.som_cols

    def get_som_weights(self) -> torch.Tensor:
        """
        Returns the current weights of the SOM
        :return: Tensor of shape ``(som_rows * som_cols, num_patches * embed_dim)``
        """
        return self.som_weights

    def get_weight_of_node(self, flat_idx: int) -> torch.Tensor:
        """
        Returns the weight of the node in the SOM at the given index
        :param flat_idx: Flat index of the neuron
        :return: Weight Tensor
        """
        return self.som_weights[flat_idx]

    def find_dissimilar_neighbour(self,
                                  e_index: tuple[int,int],
                                  e_index_flat: int) -> tuple[int,int]:
        """
        Method to find the most dissimilar neighbour in a rectangular grid to the neuron unit e at given index
        :param e_index: Index on the 2d grid of the neuron unit e
        :param e_index_flat: Flat index of the neuron unit e in the Parameter
        :return: Index of the most dissimilar neighbour d to the unit e
        """
        e_weight = self.get_weight_of_node(e_index_flat)
        r, c = e_index
        max_dist = -1.0
        d_idx = None

        coords_neighbours = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for rn, cn in coords_neighbours:
            if 0 <= cn < self.current_col_num and 0 <= rn < self.current_row_num:
                # calculate flat index of neighbour to get the weight
                flat_idx_n = rn * self.current_col_num + cn
                neighbor_weight = self.get_weight_of_node(flat_idx_n)

                dist = 1 - F.cosine_similarity(e_weight, neighbor_weight, dim=0)
                if dist > max_dist:
                    max_dist = dist
                    d_idx = (rn, cn)

        return d_idx

    def add_col_between(self, col1: int, col2: int):
        """
        Method which inserts a new column between given column indices
        :param col1: Index of the first column
        :param col2: Index of the second column
        """
        flat_weights = self.som_weights.data
        grid_weights = flat_weights.view(self.current_row_num, self.current_col_num, self.som_dim)

        insert_idx = max(col1, col2)

        # calculate the weights of new column as a mean of neighbours
        col_left = grid_weights[:, col1:col1+1, :]
        col_right = grid_weights[:, col2:col2+1, :]
        new_col = (col_left + col_right) / 2

        part_left = grid_weights[:, :insert_idx, :]
        part_right = grid_weights[:, insert_idx:, :]
        new_grid = torch.cat([part_left, new_col, part_right], dim=1)

        self.current_col_num += 1
        self.som_weights = nn.Parameter(new_grid.reshape(-1, self.som_dim))

    def add_row_between(self, row1: int, row2: int):
        """
        Method which inserts a new row between given row indices
        :param row1: Index of the first row
        :param row2: Index of the second row
        """
        flat_weights = self.som_weights.data
        grid_weights = flat_weights.view(self.current_row_num, self.current_col_num, self.som_dim)

        insert_idx = max(row1, row2)

        # calculate the weights of new row as a mean of neighbours
        row_top = grid_weights[row1:row1+1, :, :]
        row_bottom = grid_weights[row2:row2+1, :, :]
        new_row = (row_top + row_bottom) / 2

        part_top = grid_weights[:insert_idx, :, :]
        part_bottom = grid_weights[insert_idx:, :, :]
        new_grid = torch.cat([part_top, new_row, part_bottom], dim=0)

        self.current_row_num += 1
        self.som_weights = nn.Parameter(new_grid.reshape(-1, self.som_dim))

    def grow(self, unit_error_matrix: np.ndarray):
        """
        Method which grows the grid by one row or one column
        :param unit_error_matrix: Error matrix to find the neuron unit e with the highest error
        """
        e_index_flat = np.argmax(unit_error_matrix)
        e_index = np.unravel_index(e_index_flat, unit_error_matrix.shape)

        d_index = self.find_dissimilar_neighbour(e_index, e_index_flat)

        if d_index is None:
            print(f"---------------------No neighbour found for {e_index}---------------")
            return

        er, ec = e_index
        dr, dc = d_index

        if er == dr:  # same row, adding column between their cols
            self.add_col_between(ec, dc)
        elif ec == dc:  # same col, adding row between their rows
            self.add_row_between(er, dr)
        else:
            raise ValueError("e_unit and d_unit not adjacent")

    def calculate_unit_errors(self,
                              loader: torch.utils.data.DataLoader,
                              device: torch.device) -> np.ndarray[np.ndarray[float]]:
        """
        Method calculating error for each neuron unit
        :param loader: Dataloader
        :param device: The torch device
        :return: Error for each unit
        """

        unit_errors = np.zeros((self.current_row_num, self.current_col_num))

        total_samples = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                total_samples += images.shape[0]

                latent = self.encoder(images)
                patches = latent[:, 1:, :]
                som_input = patches.reshape(patches.shape[0], -1)

                # find bmu for batch
                dists = cosine_distance_torch(self.get_som_weights(), som_input)

                # get min distance and flat index in the batch
                # min_dists: minimal error value for each image
                # flat_indices: flatted indices of winning node (0,n-1)
                min_dists, flat_indices = torch.min(dists, dim=1)
                min_dists = min_dists.cpu().numpy()
                flat_indices = flat_indices.cpu().numpy()

                # e.g. flat index 7 in 5 col grid -> row 1, col 2
                row_indices = flat_indices // self.current_col_num
                col_indices = flat_indices % self.current_col_num

                # if multiple images in same batch hit same neuron, value is added only ones, therefore have to use np.add.at
                #unit_errors[row_indices, col_indices] += min_dists
                np.add.at(unit_errors, (row_indices, col_indices), min_dists)

        return unit_errors

    def start_growth(self, loader: torch.utils.data.DataLoader, device: torch.device):
        """
        Method which starts growing grid by one row or one column
        :param loader: Dataloader
        :param device: The torch device
        """
        self.eval()
        unit_errors = self.calculate_unit_errors(loader, device)

        self.grow(unit_errors)
        print(f"Current grid size: ({self.current_row_num}, {self.current_col_num})")
        self.to(device)
        self.train()
