# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpmath.libmp import normalize


def extract_patches(featmap_bchw, patch_size=3, stride=3):
    """
    Patch feature map (B, C, H, W) => (B, N, C, patch_size, patch_size)
    N = (H/stride)*(W/stride) (No overlapping)
    """
    unfolded = F.unfold(featmap_bchw, kernel_size=patch_size, stride=stride)
    B, total_size, N = unfolded.shape  # total_size = C*patch_size*patch_size
    C = featmap_bchw.size(1)
    # reshape => (B, N, C, patch_size, patch_size)
    patches = unfolded.transpose(1, 2).reshape(B, N, C, patch_size, patch_size)
    return patches


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

        self.in_channels = in_channels
        self.patch_size = patch_size


        if self.in_channels == 1:
            # [vocab_size, in_channels, patch_size, patch_size]

            tensor = torch.linspace(0.0000, 0.6560, steps=81 * 81).reshape(1, 1, 81, 81)
            tensor_patch = extract_patches(tensor, patch_size=patch_size, stride=patch_size).squeeze(0)
            self.vocabulary = nn.Parameter(
                tensor_patch
            )

            self.vocab_size = 729



        else:
            N=729
            composition_matrix = torch.zeros(1, N, 729)
            for i in range(N):
                current_cm = torch.zeros(N)
                current_cm[i] = 1
                composition_matrix[:, i, :] = current_cm
            composition_matrix = composition_matrix.view(1, 27, 27, 729)
            composition_matrix = composition_matrix.permute(0, 3, 1, 2)

            composition_matrix = extract_patches(
                composition_matrix, patch_size=self.patch_size, stride=self.patch_size
            ).squeeze(0)
            self.vocabulary = nn.Parameter(composition_matrix)

            self.vocab_size = 81

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

        # Convert vocabulary flatten into (vocab_size, C*H*W)
        vocab = self.vocabulary.unsqueeze(0).unsqueeze(0)

        # TODO: Remove flatting machanism
        # MSE
        mse = torch.mean((x - vocab) ** 2, dim=(-3, -2, -1))  # (B, N, vocab_size)

        composition_matrix = torch.zeros(B, N, self.vocab_size).to(x.device)
        for i in range(N):
            current_cm = torch.zeros(self.vocab_size).to(x.device)
            current_cm[i] = 1
            composition_matrix[:, i, :] = current_cm

        return composition_matrix
