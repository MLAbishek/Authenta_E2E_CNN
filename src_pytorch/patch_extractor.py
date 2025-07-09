import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchExtractorLayer(nn.Module):
    def __init__(self, tile_size=256, tile_stride=192, **kwargs):
        super().__init__()
        self.tile_size = tile_size
        self.tile_stride = tile_stride

    def forward(self, inputs):
        # inputs shape: (batch_size, C, H, W) - PyTorch uses channel-first format
        # But if your input is (batch_size, H, W, C), we need to permute first
        if inputs.dim() == 4 and inputs.shape[1] > inputs.shape[3]:
            # Assume input is (batch_size, H, W, C) - convert to (batch_size, C, H, W)
            inputs = inputs.permute(0, 3, 1, 2)

        batch_size, channels, height, width = inputs.shape

        # Extract patches using unfold operation
        # unfold(dimension, size, step) extracts sliding windows
        patches = inputs.unfold(2, self.tile_size, self.tile_stride)  # Unfold height
        patches = patches.unfold(3, self.tile_size, self.tile_stride)  # Unfold width

        # patches shape: (batch_size, channels, num_patches_h, num_patches_w, tile_size, tile_size)
        batch_size, channels, num_patches_h, num_patches_w, tile_h, tile_w = (
            patches.shape
        )
        total_patches = num_patches_h * num_patches_w

        # Reshape to (batch_size * total_patches, channels, tile_size, tile_size)
        patches = patches.permute(
            0, 2, 3, 1, 4, 5
        )  # (batch_size, num_patches_h, num_patches_w, channels, tile_size, tile_size)
        patches = patches.contiguous().view(
            batch_size * total_patches, channels, self.tile_size, self.tile_size
        )

        # If you want output in (batch_size * total_patches, tile_size, tile_size, channels) format
        # to match TensorFlow's channel-last format, uncomment the next line:
        # patches = patches.permute(0, 2, 3, 1)

        return patches, batch_size, total_patches
