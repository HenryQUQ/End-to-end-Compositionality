# pipeline.py
import torch
import torch.nn as nn

from models.composition.composition_layer import CompositionalLayer
from torch.nn import functional as F
# from pipelines.utils.patch_related import extract_patches, fold_patches

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

def fold_patches(patches, out_h, out_w, patch_size=3, stride=3):
    """
    Restore patches (B, N, C, patch_size, patch_size) to (B, C, out_h, out_w)
    """
    B, N, C, H, W = patches.shape
    # (B, N, C*H*W) => (B, C*H*W, N)
    patches_reshape = patches.reshape(B, N, C * H * W).transpose(1, 2)
    # Use Fold
    folded = F.fold(
        patches_reshape,
        output_size=(out_h, out_w),
        kernel_size=patch_size,
        stride=stride,
    )
    return folded  # (B, C, out_h, out_w)

if __name__ == "__main__":
    a = torch.rand(1, 10, 81, 81)
    patches = extract_patches(a, patch_size=3, stride=3)
    print(patches.shape)
    folded = fold_patches(patches, out_h=81, out_w=81)
    print(folded.shape)
    print((a == folded).all())


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
        self._visualize_composition_matrix(info_list, layer_number=1)
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
                "featmap_bchw": featmap_bchw,
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
        # TODO： limit the channel channel sum to 1 and 0-1
        # From top to the bottom
        # info_list[-1] is the top layer
        reconstructed_info = [None for _ in range(len(self.layers))]
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            info = info_list[i]

            # Get the composition matrix that *this* layer produced
            cm = info["composition_matrix"]  # (B, N, vocab_size_i)

            # Combine cm with the layer's vocabulary to get the raw patches
            patches = self._combine_cmatrix(cm, layer.vocabulary)
            # patches.shape = (B, N, in_channels_of_this_layer, patch_size, patch_size)

            # We know the shape of the feature map that went into this layer
            B_in, C_in, H_in, W_in = info["in_featmap_shape"]

            # Reshape patches so we can fold them:
            #   patches for fold => (B, C_in*patch_size*patch_size, N)
            B, N, C, ph, pw = patches.shape
            patches_for_fold = patches.view(B, N, C * ph * pw).permute(0, 2, 1)
            # Now fold to get the layer’s input feature map
            reconstructed_current = F.fold(
                patches_for_fold,
                output_size=(H_in, W_in),
                kernel_size=self.patch_size,
                stride=self.stride
            )

            reconstructed_info[i] = reconstructed_current

        return reconstructed_info

    def _combine_cmatrix(self, cm_current, vocabulary):
        """

        - cm_current.shape = (B, N, vocab_size_i)
        - vocabulary.shape = (vocab_size_i, in_channels, 3, 3)

        Return:
        - cm_next: (B, N, in_channels, 3, 3)
        """
        B, N, V = cm_current.shape
        V2, C, H, W = vocabulary.shape
        assert V == V2, "Vocabulary size mismatch."


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

    def _visualize_vocabulary(self):
        voc_list =[]
        for i in reversed(range(len(self.layers))):
            current_v = self.layers[i].vocabulary

            cm_current = current_v
            B, C, H, W = cm_current.shape

            for j in reversed(range(i-1)):
                layer = self.layers[j]
                cm_next = self._combine_cmatrix(
                    cm_current, layer.vocabulary
                )

                B, N_i, C, H, W = cm_next.shape
                cm_next = cm_next.view(B, N_i, C, H * W)  # (B, N_i, C, 9)
                cm_next = cm_next.permute(0, 1, 3, 2)  # (B, N_i, 9, C)
                cm_next = cm_next.reshape(B, N_i * 9, C)  # (B, N_i*9, C)
                cm_current = cm_next


            patches_reshape = cm_current.view(B, -1, 1 * 3 * 3).transpose(1, 2)
            final_image = F.fold(
                patches_reshape, output_size=(3**(i-1), 3**(i-1)), kernel_size=3, stride=3)
            voc_list.append(final_image)

        return voc_list

    def _visualize_composition_matrix(self, composition_matrix, layer_number:int):

        cm_current = composition_matrix[-1]["composition_matrix"]
        # cm_current = composition_matrix
        for current_layer_number in range(layer_number, -1, -1):
            current_layer = self.layers[current_layer_number]
            current_vocabulary = current_layer.vocabulary
            cm_next = self._combine_cmatrix(
                cm_current, current_vocabulary
            )

            B, N_i, C, H, W = cm_next.shape

            cm_next = fold_patches(cm_next, out_h=int((N_i*H*W)**0.5), out_w=int((N_i*H*W)**0.5))

            if current_layer_number == 0:
                cm_current = cm_next
                break
            cm_next = cm_next.permute(0, 2, 3, 1).contiguous()

            B, H, W, C = cm_next.shape

            cm_next = cm_next.view(B, H*W, C)

            cm_current = cm_next
        return cm_current








