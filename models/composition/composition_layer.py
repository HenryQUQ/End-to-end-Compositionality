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

        # flatten x -> (B, N, 1, C*H*W)
        x_flat = x.view(B, N, 1, C * H * W)

        # flatten vocabulary -> (vocab_size, C*H*W)
        vocab_flat = self.vocabulary.view(self.vocab_size, -1)  # (vocab_size, C*H*W)
        # reshape -> (1, 1, vocab_size, C*H*W)
        vocab_flat = vocab_flat.unsqueeze(0).unsqueeze(0)

        # Dot Product -> (B, N, vocab_size)
        dot = (x_flat * vocab_flat).sum(dim=-1)

        # Cosine Similarity
        x_norm = x_flat.norm(dim=-1, keepdim=False)  # (B, N, 1)
        vocab_norm = vocab_flat.norm(dim=-1, keepdim=False)  # (1, 1, vocab_size)
        cos_sim = dot / (x_norm * vocab_norm + 1e-8)

        # softmax
        composition_matrix = F.softmax(cos_sim, dim=-1)

        return composition_matrix
