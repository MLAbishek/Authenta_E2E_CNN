import tensorflow as tf
from tensorflow.keras import layers  # type:ignore


class PatchExtractorLayer(layers.Layer):
    def __init__(self, tile_size=256, tile_stride=192, **kwargs):
        super().__init__(**kwargs)
        self.tile_size = tile_size
        self.tile_stride = tile_stride

    def call(self, inputs):
        # inputs shape: (batch_size, H, W, 3)
        # E2E style: overlapping tiles with stride < tile_size

        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.tile_size, self.tile_size, 1],
            strides=[1, self.tile_stride, self.tile_stride, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        batch_size = tf.shape(inputs)[0]
        num_patches_h = tf.shape(patches)[1]
        num_patches_w = tf.shape(patches)[2]
        total_patches = num_patches_h * num_patches_w
        channels = tf.shape(inputs)[-1]

        # Reshape patches to (batch_size * num_patches, tile_size, tile_size, channels)
        patches = tf.reshape(
            patches,
            (batch_size * total_patches, self.tile_size, self.tile_size, channels),
        )

        return patches, batch_size, total_patches
