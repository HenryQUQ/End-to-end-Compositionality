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

        # Intermediate information for reconstruction
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

            # Save intermediate information for reconstruction
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
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            info = info_list[i]

            # Get: composition_matrix, patches_shape, in_featmap_shape
            cm = info["composition_matrix"]  # (B, N, vocab_size_i)
            B, N, in_ch, ph, pw = info["patches_shape"]
            (B2, in_ch2, H2, W2) = info["in_featmap_shape"]

            # From composition matrix and vocabulary to compute patches
            #  composition_matrix: (B, N, vocab_size_i)
            #  vocabulary: (vocab_size_i, in_ch, ph, pw)
            # => patches_recon: (B, N, in_ch, ph, pw)
            patches_recon = self._combine_patches(cm, layer.vocabulary)

            # fold => (B, in_ch, H2, W2)
            featmap_recon = fold_patches(
                patches_recon,
                out_channels=in_ch,
                out_h=H2,
                out_w=W2,
                patch_size=self.patch_size,
                stride=self.stride,
            )

            # use featmap_recon as lower layer's featmap
            if i > 0:
                # Save featmap_recon to info_list[i-1],
                # For the next layer's reconstruction
                info_list[i - 1]["recon_featmap"] = featmap_recon
            else:
                # i==0, return the final reconstructed image
                return featmap_recon

        # Should not reach here
        return None

    def _combine_patches(self, cm, vocabulary):
        """
        Use composition matrix and vocabulary to compute patches,
        input:
         cm: (B, N, vocab_size)
         vocabulary: (vocab_size, in_ch, ph, pw)
        return:
         patches_recon: (B, N, in_ch, ph, pw)
        """
        B, N, V = cm.shape
        V2, in_ch, ph, pw = vocabulary.shape
        assert V == V2, "vocab_size mismatch"

        # expand => (B, N, V, 1, 1, 1) and (1, 1, V, in_ch, ph, pw)
        cm_expanded = cm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        vocab_expanded = vocabulary.unsqueeze(0).unsqueeze(0)

        # Times => (B, N, V, in_ch, ph, pw)
        patch_combined = cm_expanded * vocab_expanded
        # Sum the V channel => (B, N, in_ch, ph, pw)
        patches_recon = patch_combined.sum(dim=2)

        return patches_recon
