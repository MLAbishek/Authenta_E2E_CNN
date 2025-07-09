import tensorflow as tf
import numpy as np
from pathlib import Path
import os
from src.E2EModel import E2EDetectionModel


def load_image(image_path):
    """Load and preprocess image"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    # Set shape explicitly to fix the resize issue
    image.set_shape([None, None, 3])
    return image


def create_dataset(data_dir, batch_size=8, image_size=(512, 512)):
    """Create dataset from folder structure (original/duplicate folders)"""

    # Get image paths and labels from folder structure
    image_paths = []
    labels = []

    # Process original images (label = 0)
    original_dir = os.path.join(data_dir, "original")
    if os.path.exists(original_dir):
        for image_file in os.listdir(original_dir):
            if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(original_dir, image_file))
                labels.append(0)  # 0 for original

    # Process duplicate images (label = 1)
    duplicate_dir = os.path.join(data_dir, "duplicate")
    if os.path.exists(duplicate_dir):
        for image_file in os.listdir(duplicate_dir):
            if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(duplicate_dir, image_file))
                labels.append(1)  # 1 for duplicate

    print(f"Found {len([l for l in labels if l == 0])} original images")
    print(f"Found {len([l for l in labels if l == 1])} duplicate images")
    print(f"Total: {len(image_paths)} images")

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def preprocess_fn(image_path, label):
        image = load_image(image_path)
        # Resize to fixed size
        image = tf.image.resize(image, image_size)
        return image, label

    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def train_model(train_dir, val_dir, epochs=10, batch_size=4, learning_rate=0.001):
    """Simple training function"""

    # Create datasets
    train_dataset = create_dataset(train_dir, batch_size)
    val_dataset = create_dataset(val_dir, batch_size)

    # Create model
    model = E2EDetectionModel(tile_size=256, tile_stride=128, num_classes=2)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5, monitor="val_loss", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, monitor="val_loss"
        ),
    ]

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


# Example usage
if __name__ == "__main__":
    # Update these paths to your data
    TRAIN_DIR = r"D:\Data\train"  # Should contain 'original' and 'duplicate' folders
    VAL_DIR = r"D:\Data\test"  # Should contain 'original' and 'duplicate' folders

    # Expected folder structure:
    # train/
    # ├── original/
    # │   ├── image1.jpg
    # │   ├── image2.jpg
    # │   └── ...
    # └── duplicate/
    #     ├── image3.jpg
    #     ├── image4.jpg
    #     └── ...

    print("Starting training...")
    print("Target classes: 0 = Original, 1 = Duplicate")
    model, history = train_model(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        epochs=5,
        batch_size=32,
        learning_rate=0.001,
    )

    print("Training completed!")
    print(f"Final validation accuracy: {max(history.history['val_accuracy']):.4f}")

    # Save final model
    model.save_weights("final_model.weights.h5")
    print("Model saved as 'final_model.weights.h5'")
