# pipeline.py
import torch
import torch.nn as nn

from models.composition.composition_layer import CompositionalLayer
from torch.nn import functional as F
from pipelines.utils.patch_related import extract_patches, fold_patches


class CompositionalPipeline(nn.Module):
    """
    forward(image):
      - Patch Image => composition matrix => reshape to feature map => Repeat to the next layer
      - Save intermediate information for reconstruction
    reconstruct():
      - From top to the bottom，each layer uses composition matrix + vocabulary to get patch => fold to feature map of previous layer
      - Return the final reconstructed image
    """

    def __init__(
        self, in_channels=1, vocab_sizes=[10, 5], patch_size=3, stride=3, image_size=81
    ):
        super().__init__()
        self.in_channels = in_channels
        self.vocab_sizes = vocab_sizes
        self.patch_size = patch_size
        self.stride = stride
        self.image_size = image_size

        self.layers = nn.ModuleList()
        current_in_channels = in_channels
        for vs in vocab_sizes:
            layer = CompositionalLayer(
                vocab_size=vs, in_channels=current_in_channels, patch_size=patch_size
            )
            self.layers.append(layer)
            current_in_channels = vs

    def forward(self, images):
        final_feat, info_list = self.compose(images)
        reconstructed = self.reconstruct(info_list)
        return final_feat, info_list, reconstructed

    def compose(self, images):
        """
        images: (B, in_channels, image_size, image_size)
        Return:
          final_featmap: (B, H_final, W_final, vocab_sizes[-1])
          info_list: The intermediate information for reconstruction
        """
        B, C, H, W = images.shape
        assert (
            H == self.image_size and W == self.image_size
        ), "Input image size must match self.image_size."

        # Intermediate information
        info_list = []

        # Current feature map (bchw)
        featmap_bchw = images

        for i, layer in enumerate(self.layers):
            in_ch = featmap_bchw.size(1)
            # 1) Get patch => (B, N, in_ch, 3, 3)
            patches = extract_patches(
                featmap_bchw, patch_size=self.patch_size, stride=self.stride
            )
            B, N, in_ch, ph, pw = patches.shape
            # 2) Composition Layer => (B, N, vocab_size_i)
            cm = layer(patches)
            vocab_size_i = cm.size(-1)

            # 3) composition matrix => 2D feature map => (B, H_out, W_out, vocab_size_i)
            H_out, W_out = H // self.stride, W // self.stride
            featmap_2d = cm.view(B, H_out, W_out, vocab_size_i)

            # Save intermediate information
            info = {
                "layer_idx": i,
                "patches_shape": (B, N, in_ch, ph, pw),
                "in_featmap_shape": (B, in_ch, H, W),
                "out_featmap_shape": (B, vocab_size_i, H_out, W_out),
                "composition_matrix": cm,
            }
            info_list.append(info)

            # Update Feature Map shape for the next layer
            # permute => (B, vocab_size_i, H_out, W_out)
            featmap_bchw = featmap_2d.permute(0, 3, 1, 2).contiguous()

            # Update H, W
            H, W = H_out, W_out

        # Final featmap => (B, H, W, vocab_size[-1])
        final_featmap = featmap_bchw.permute(0, 2, 3, 1).contiguous()

        return final_featmap, info_list

    def reconstruct(self, info_list):
        """
        Based on info_list, reverse the forward process, from top to the bottom, return (B, in_channels, image_size, image_size) image reconstruction
        """
        # From top to the bottom
        # info_list[-1] is the top layer
        top_idx = len(self.layers) - 1

        cm_current = info_list[top_idx]["composition_matrix"]

        # We only use the top layer's composition matrix.
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            info = info_list[i]

            cm_next = self._combine_cmatrix(
                cm_current, layer.vocabulary, normalize=False
            )

            B, N_i, C, H, W = cm_next.shape
            cm_next = cm_next.view(B, N_i, C, H * W)  # (B, N_i, C, 9)
            cm_next = cm_next.permute(0, 1, 3, 2)  # (B, N_i, 9, C)
            cm_next = cm_next.reshape(B, N_i * 9, C)  # (B, N_i*9, C)
            cm_current = cm_next

        B0, C0, H0, W0 = info_list[0]["in_featmap_shape"]  # Original image shape

        patches_reshape = cm_current.view(B, -1, C0 * 3 * 3).transpose(1, 2)
        final_image = F.fold(
            patches_reshape, output_size=(H0, W0), kernel_size=3, stride=3
        )
        return final_image

    def _combine_cmatrix(self, cm_current, vocabulary, normalize=True):
        """

        - cm_current.shape = (B, N, vocab_size_i)
        - vocabulary.shape = (vocab_size_i, in_channels, 3, 3)

        Return:
        - cm_next: (B, N, in_channels, 3, 3), 表示下一层的 construction matrix
        """
        B, N, V = cm_current.shape
        V2, C, H, W = vocabulary.shape
        assert V == V2, "Vocabulary size mismatch."

        if normalize:
            cm_current = cm_current / (cm_current.sum(dim=-1, keepdim=True) + 1e-8)

        # (B, N, V, 1, 1, 1) vs (1, 1, V, C, H, W)
        cm_expanded = (
            cm_current.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # (B,N,V,1,1,1)

        vocab_expanded = vocabulary.unsqueeze(0).unsqueeze(0)  # (1,1,V,C,H,W)

        # => (B, N, V, C, H, W)
        patches = cm_expanded * vocab_expanded
        # sum over V => (B, N, C, H, W)
        cm_next = patches.sum(dim=2)

        return cm_next
