import torch
import torch.nn as nn


class StatisticalAggregatorLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, features, batch_size, num_patches, targets=None):
        # features shape: (batch_size * num_patches, H, W, C)
        feature_shape = features.shape
        feat_h, feat_w, feat_c = feature_shape[1], feature_shape[2], feature_shape[3]

        # Reshape to group by original batch
        features = features.view(batch_size, num_patches, feat_h, feat_w, feat_c)

        # Apply statistical pooling operations across all patches at once
        # features shape: (batch_size, num_patches, H, W, C)

        # Apply 4 statistical operations across spatial dimensions (H, W)
        max_pool = torch.max(features, dim=2)[0]  # (batch_size, num_patches, W, C)
        max_pool = torch.max(max_pool, dim=2)[0]  # (batch_size, num_patches, C)

        min_pool = torch.min(features, dim=2)[0]  # (batch_size, num_patches, W, C)
        min_pool = torch.min(min_pool, dim=2)[0]  # (batch_size, num_patches, C)

        avg_pool = torch.mean(features, dim=[2, 3])  # (batch_size, num_patches, C)

        # Corrected squared pooling: average of squared features
        squared_pool = torch.mean(
            features**2, dim=[2, 3]
        )  # (batch_size, num_patches, C)

        # Concatenate all statistics for each patch
        # Shape: (batch_size, num_patches, 4*C)
        all_patch_features = torch.cat(
            [max_pool, min_pool, avg_pool, squared_pool], dim=-1
        )

        # Apply global aggregation across all patches
        global_max = torch.max(all_patch_features, dim=1)[0]  # (batch_size, 4*C)
        global_min = torch.min(all_patch_features, dim=1)[0]  # (batch_size, 4*C)
        global_avg = torch.mean(all_patch_features, dim=1)  # (batch_size, 4*C)

        # Final concatenation
        final_features = torch.cat([global_max, global_min, global_avg], dim=-1)

        return final_features
