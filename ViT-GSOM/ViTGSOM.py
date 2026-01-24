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
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, som_weights, grid_coords, sigma):
        # distance for all samples in batch, shape (batch, Num_Units)
        dists = cosine_distance_torch(som_weights, input_vectors)

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

class ViTSOMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mseLoss = nn.MSELoss()
        self.somLoss = SomLoss()

    def forward(self, original_img, reconstructed, latent_vectors, som_weights, grid_coords, sigma, current_lamda):
        l_nn = self.mseLoss(original_img, reconstructed)

        # latent vector shape: (batch, sequence of patches + cls, embed_dim), cls not needed for SOM, only patches
        patches = latent_vectors[:, 1:, :] 
        
        som_input = patches.reshape(patches.shape[0], -1)
        l_som = self.somLoss(som_input, som_weights, grid_coords, sigma)
        l_total = (current_lamda * l_som) + l_nn
        return l_total, l_nn, l_som        # Eq. 6


def unpatch(x, patch_size=4, channels=1):
    # E.g. (8, 49, 3*4*4): batch of 8, 7x7 grid, num_of_channels * patch_size * patch_size
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
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # convolution with the stride size same as patch size -> no overlapping
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
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

    def __init__(self, embed_dim, mlp_dim, dropout):
        super().__init__()
        self.dense_1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
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

    def __init__(self, img_size=28, patch_size=4, in_channels=1,
                 embed_dim=16, depth=4, num_heads=2, mlp_dim=64):
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

    def forward(self, x):
        # create patches
        B = x.shape[0]
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add Positional Embedding
        x = x + self.pos_embed

        # apply self attention layers and mlp
        for block in self.blocks:
            x = block(x)

        x = self.ln1(x)
        return x



class ViTDecoder(nn.Module):
    def __init__(self, num_patches, patch_size=4, output_dim = 1, embed_dim=16, depth=2, num_heads=2, mlp_dim=64):
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
        self.head = nn.Linear(embed_dim, self.pixels_per_patch)

    def forward(self, x):
        # positional embeddings in latent space
        x = x + self.pos_embed

        # applying self attention layers and mlp
        for block in self.blocks:
            x = block(x)

        x = self.ln1(x)
        # removing CLS token (e.g. from PatchEmbedding (8, 50, 64) -> (8, 49, 64))
        x = x[:, 1:, :]

        # projection back to pixel space
        x = self.head(x)  # e.g. (8, 49, 64) -> (8, 49, 16): (B, grid 7x7, pixels per patch)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, img_size=28, patch_size=4, num_of_channels=1, embed_dim=16, enc_depth=4,
                 dec_depth=2, num_heads=2, mlp_dim=64, som_rows = 2, som_cols = 2, spread_factor = 0.5):
        super().__init__()

        assert img_size % patch_size == 0, f"Image size ({img_size}) must be divisible by patch size ({patch_size})."

        self.num_of_channels = num_of_channels
        self.patch_size = patch_size

        self.encoder = ViTEncoder(img_size, patch_size, num_of_channels, embed_dim, enc_depth, num_heads, mlp_dim)
        self.num_of_patches = (img_size // patch_size) ** 2
        self.decoder = ViTDecoder(self.num_of_patches, patch_size, num_of_channels, embed_dim, dec_depth, num_heads, mlp_dim)

        self.current_row_num = som_rows
        self.current_col_num = som_cols
        self.spread_factor = spread_factor
        self.mqe0 = None
        self.som_dim = self.num_of_patches * embed_dim
        self.som_weights = nn.Parameter(torch.randn(self.current_row_num * self.current_col_num, self.som_dim))

    def forward(self, x):
        latent = self.encoder(x)
        patched_output = self.decoder(latent)
        output = unpatch(patched_output, self.patch_size, self.num_of_channels)
        return output, latent

    def get_sigma(self):
        return np.ceil(min(self.current_row_num, self.current_col_num) / 2)

    def get_som_shape(self):
        return self.current_row_num, self.current_col_num

    def get_som_weights(self):
        return self.som_weights

    def get_weight_of_node(self, flat_idx):
        return self.som_weights[flat_idx]

    def calculate_mqe0(self, loader, device):
        self.eval()
        all_latent = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)

                _, latent = self.encoder(images)
                patches = latent[:, 1:, :]
                # (batch, 784)
                flat_latent = patches.reshape(patches.shape[0], -1)
                all_latent.append(flat_latent.cpu())

        # (n_samples, 784)
        data_latent = torch.cat(all_latent, dim=0)
        # centroid as a mean of latent, shape (1,784)
        centroid_latent = torch.mean(data_latent, dim=0, keepdim=True)

        data_latent = data_latent.to(device)
        centroid_latent = centroid_latent.to(device)

        # mqe0 as distance from all latent to centroid latent - latent variance
        dists = self.cosine_distance_torch(centroid_latent, data_latent)
        mqe0 = torch.mean(dists).item()

        print(f"Latent variance (mqe0): {mqe0}")
        print(f"Spread Factor: {self.spread_factor}")
        print(f"Threshold, mqe0 * spread_factor: {mqe0 * self.spread_factor}")
        return mqe0


    def find_dissimilar_neighbour(self, e_index, e_index_flat):
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

    def add_col_between(self, col1, col2):
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


    def add_row_between(self, row1, row2):
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

    def grow(self, unit_error_matrix):
        e_index_flat = np.argmax(unit_error_matrix)
        e_index = np.unravel_index(e_index_flat, unit_error_matrix.shape)

        d_index = self.find_dissimilar_neighbour(e_index, e_index_flat)

        if d_index is None:
            print(f"---------------------No neighbour found for {e_index}---------------")
            return

        er, ec = e_index
        dr, dc = d_index

        if er == dr:  # same row
            self.add_col_between(ec, dc)
        elif ec == dc:  # same col
            self.add_row_between(er, dr)
        else:
            raise ValueError("e_unit and d_unit not adjacent")

    def calculate_unit_errors(self, loader, device):

        unit_errors = np.zeros((self.current_row_num, self.current_col_num))
        unit_hits = np.zeros((self.current_row_num, self.current_col_num))

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)

                _, latent = self.encoder(images)
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
                #unit_hits[row_indices, col_indices] += 1
                np.add.at(unit_hits, (row_indices, col_indices), 1)

        unit_hits_mask = unit_hits > 0
        active_units_count = np.sum(unit_hits_mask)

        # Global MQE = Total Error / number of active units in the GSOM
        total_error = np.sum(unit_errors)
        global_mqe = total_error / active_units_count if active_units_count > 0 else 0

        return unit_errors, global_mqe

    def check_growth(self, loader, device):
        if self.mqe0 is None:
            self.mqe0 = self.calculate_mqe0(loader, device)

        self.eval()
        unit_errors, global_mqe = self.calculate_unit_errors(loader, device)
        output = False

        growth_threshold = self.spread_factor * self.mqe0
        if global_mqe > growth_threshold:
            print(f"MQE {global_mqe:.4f} > Threshold {growth_threshold}. Growing")
            self.grow(unit_errors)
            self.to(device)
            output = True

        self.train()
        return output