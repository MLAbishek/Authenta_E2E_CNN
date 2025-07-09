import torch
import torch.nn as nn
import timm

# Note: PyTorch doesn't have Xception in torchvision by default
# You'll need to install timm: pip install timm


class FeatureExtractionLayer(nn.Module):
    def __init__(self, trainable=False, **kwargs):
        super().__init__()

        self.backbone = timm.create_model("xception", pretrained=True, num_classes=0)

        # Set trainable parameters
        if not trainable:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        # inputs shape: (batch_size * num_patches, 3, tile_size, tile_size)
        # Note: PyTorch uses channels-first format, TensorFlow uses channels-last

        # Process all patches through Xception
        features = self.backbone(inputs)

        return features
