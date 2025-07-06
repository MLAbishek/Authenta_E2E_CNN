from src.agregator import StatisticalAggregatorLayer
from tensorflow.keras import layers  # type:ignore
import tensorflow as tf


class E2EDetectionModel(tf.keras.Model):
    def __init__(self, tile_size=256, tile_stride=192, num_classes=2, **kwargs):
        super().__init__(**kwargs)

        # Import the layers (assuming they're in the same directory)
        from src.patch_extractor import PatchExtractorLayer
        from src.feature_extractor import FeatureExtractionLayer

        self.patch_extractor = PatchExtractorLayer(
            tile_size=tile_size, tile_stride=tile_stride
        )
        self.feature_extractor = FeatureExtractionLayer()
        self.aggregator = StatisticalAggregatorLayer()

        self.classifier = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(num_classes),
            ]
        )

    def call(self, inputs, training=None):
        # Extract patches
        patches, batch_size, num_patches = self.patch_extractor(inputs)

        # Extract features
        features = self.feature_extractor(patches)

        # Aggregate features
        aggregated_features = self.aggregator(features, batch_size, num_patches)

        # Classify
        logits = self.classifier(aggregated_features)

        return logits
