import tensorflow as tf
from tensorflow.keras import layers  # type:ignore
from tensorflow.keras.applications import Xception  # type:ignore


class FeatureExtractionLayer(layers.Layer):
    def __init__(self, trainable=False, **kwargs):
        super().__init__(**kwargs)
        # Use Xception as backbone (same as E2E)
        self.backbone = Xception(include_top=False, weights="imagenet")
        self.backbone.trainable = trainable

    def call(self, inputs):
        # inputs shape: (batch_size * num_patches, tile_size, tile_size, 3)
        # Process all patches through Xception
        features = self.backbone(inputs, training=False)

        return features
