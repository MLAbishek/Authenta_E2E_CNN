import torch
import torch.nn as nn


class E2EDetectionModel(nn.Module):
    def __init__(self, tile_size=256, tile_stride=192, num_classes=2, **kwargs):
        super().__init__()

        # Import the layers (assuming they're in the same directory)
        from src_pytorch.patch_extractor import PatchExtractorLayer
        from src_pytorch.feature_extractor import FeatureExtractionLayer
        from src_pytorch.aggregator import StatisticalAggregatorLayer

        self.patch_extractor = PatchExtractorLayer(
            tile_size=tile_size, tile_stride=tile_stride
        )
        self.feature_extractor = FeatureExtractionLayer()
        self.aggregator = StatisticalAggregatorLayer()

        self.classifier = nn.Sequential(
            nn.Linear(None, 512),  # Input size will be determined dynamically
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
        )

        # Flag to track if classifier input size has been set
        self._classifier_initialized = False

    def _initialize_classifier(self, input_size):
        """Initialize classifier with correct input size"""
        if not self._classifier_initialized:
            self.classifier[0] = nn.Linear(input_size, 512)
            self._classifier_initialized = True

    def forward(self, inputs, training=None):
        # Extract patches
        patches, batch_size, num_patches = self.patch_extractor(inputs)

        # Extract features
        features = self.feature_extractor(patches)

        # Aggregate features
        aggregated_features = self.aggregator(features, batch_size, num_patches)

        # Initialize classifier if needed
        if not self._classifier_initialized:
            self._initialize_classifier(aggregated_features.shape[1])

        # Classify
        logits = self.classifier(aggregated_features)

        return logits
