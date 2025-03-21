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


        if self.in_channels == 1:
            # [vocab_size, in_channels, patch_size, patch_size]
            self.vocabulary = nn.Parameter(
                torch.randn(vocab_size, in_channels, patch_size, patch_size)
            )

        else:
            vocab = torch.randn(vocab_size, in_channels, patch_size, patch_size)
            vocab = torch.abs(vocab)
            vocab = vocab / vocab.sum(dim=1, keepdim=True)
            self.vocabulary = nn.Parameter(vocab)

    def forward(self, x):
        """
        x: (B, N, in_channels, patch_size, patch_size)
           B: batch size
           N: number of patches
        Return:
        composition_matrix: (B, N, vocab_size)
        """
        B, N, C, H, W = x.shape  # C=in_channels, H=patch_size, W=patch_size


        x = x.unsqueeze(2)

        vocab = self.vocabulary.unsqueeze(0).unsqueeze(0)

        # MSE
        mse = torch.mean((x - vocab) ** 2, dim=(-1, -2, -3))  # (B, N, vocab_size)

        mse_normalise = torch.tanh(mse) / torch.tanh(torch.tensor(10))

        mse_normalise = mse_normalise.clip(0, 1)
        composition_matrix = (1 - mse_normalise) ** 5

        composition_matrix = composition_matrix / composition_matrix.sum(dim=-1, keepdim=True)

        return composition_matrix
