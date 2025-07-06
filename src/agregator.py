import tensorflow as tf
from tensorflow.keras import layers  # type:ignore


class StatisticalAggregatorLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, features, batch_size, num_patches, targets=None):
        # features shape: (batch_size * num_patches, H, W, C)
        feature_shape = tf.shape(features)
        feat_h, feat_w, feat_c = feature_shape[1], feature_shape[2], feature_shape[3]

        # Reshape to group by original batch
        features = tf.reshape(
            features, (batch_size, num_patches, feat_h, feat_w, feat_c)
        )

        # Apply statistical pooling operations across all patches at once
        # features shape: (batch_size, num_patches, H, W, C)

        # Apply 4 statistical operations across spatial dimensions (H, W)
        max_pool = tf.reduce_max(features, axis=[2, 3])  # (batch_size, num_patches, C)
        min_pool = tf.reduce_min(features, axis=[2, 3])  # (batch_size, num_patches, C)
        avg_pool = tf.reduce_mean(features, axis=[2, 3])  # (batch_size, num_patches, C)

        # Corrected squared pooling: average of squared features
        squared_pool = tf.reduce_mean(
            tf.square(features), axis=[2, 3]
        )  # (batch_size, num_patches, C)

        # Concatenate all statistics for each patch
        # Shape: (batch_size, num_patches, 4*C)
        all_patch_features = tf.concat(
            [max_pool, min_pool, avg_pool, squared_pool], axis=-1
        )

        # Apply global aggregation across all patches
        global_max = tf.reduce_max(all_patch_features, axis=1)  # (batch_size, 4*C)
        global_min = tf.reduce_min(all_patch_features, axis=1)  # (batch_size, 4*C)
        global_avg = tf.reduce_mean(all_patch_features, axis=1)  # (batch_size, 4*C)

        # Final concatenation
        final_features = tf.concat([global_max, global_min, global_avg], axis=-1)

        return final_features
