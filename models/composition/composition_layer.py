# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositionalLayer(nn.Module):
    """
    Compositional Layer:
    - Trainable Vocabulary: [vocab_size, in_channels, patch_size, patch_size]
    - forward: Compute the cosine similarity between the patch -> softmax -> composition matrix
    """

    def __init__(self, vocab_size, in_channels, patch_size=3):
        """
        vocab_size: Number of vocabulary for current layer
        in_channels: Input patch channels
        patch_size: patch Size (W=H=patch_size)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.in_channels = in_channels
        self.patch_size = patch_size

        # [vocab_size, in_channels, patch_size, patch_size]
        self.vocabulary = nn.Parameter(
            torch.randn(vocab_size, in_channels, patch_size, patch_size) * 0.01
        )

    def forward(self, x):
        """
        x: (B, N, in_channels, patch_size, patch_size)
           B: batch size
           N: number of patches
        Return:
        composition_matrix: (B, N, vocab_size)
        """
        B, N, C, H, W = x.shape  # C=in_channels, H=patch_size, W=patch_size

        # Conv x flatten into (B, N, 1, C*H*W)
        x_flat = x.view(B, N, 1, C * H * W)

        # Convert vocabulary flatten into (vocab_size, C*H*W)
        vocab_flat = self.vocabulary.view(self.vocab_size, -1).unsqueeze(0).unsqueeze(0)

        # MSE
        composition_matrix = torch.mean((x_flat - vocab_flat) ** 2, dim=-1)  # (B, N, vocab_size)

        # composition_matrix = 1 / (mse + 1e-6)  # (B, N, vocab_size)
        #
        # composition_matrix = composition_matrix / composition_matrix.sum(dim=-1, keepdim=True)

        return composition_matrix
