import torch
import torch.nn as nn
import numpy as np


"""
https://tintn.github.io/Implementing-Vision-Transformer-from-Scratch/
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
https://www.geeksforgeeks.org/deep-learning/how-to-use-pytorchs-nnmultiheadattention/
"""

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # convolution with the stride size as patch size -> no overlapping
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Example: batch = 8, embed_dim=16, img_height=28, img_width=28, input_channels=1
        # x.shape: (8, 1, 28, 28)
        # 28 / 4 = 7 -> 7x7 grid
        # proj(x).shape: (8, 16, 7, 7)
        # proj(x).flatten(2): (8, 16, 7 * 7)
        # proj(x).flatten(2).transpose(1, 2): (8, 49, 16)
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
    def __init__(self, num_patches, embed_dim=16, output_dim=1, depth=2, num_heads=2, mlp_dim=64, patch_size=4):
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
        # removing CLS token (e.g. from PatchEmbedding (8, 50, 16) -> (8, 49, 16))
        x = x[:, 1:, :]

        # projection back to pixel space
        x = self.head(x)  # e.g. (8, 49, 16) since embed_dim and pixels_per_patch = 4*4 is the same
        return x