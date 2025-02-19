import torch.nn.functional as F
import math


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


def fold_patches(patches, out_channels, out_h, out_w, patch_size=3, stride=3):
    """
    Restore patches (B, N, C, patch_size, patch_size) to (B, C, out_h, out_w)
    """
    B, N, C, H, W = patches.shape
    # (B, N, C*H*W) => (B, C*H*W, N)
    patches_reshape = patches.view(B, N, C * H * W).transpose(1, 2)
    # Use Fold
    folded = F.fold(
        patches_reshape,
        output_size=(out_h, out_w),
        kernel_size=patch_size,
        stride=stride,
    )
    return folded  # (B, C, out_h, out_w)
