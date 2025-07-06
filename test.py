import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix  # type:ignore
import seaborn as sns
from src.E2EModel import E2EDetectionModel


def load_image(image_path, image_size=(512, 512)):
    """Load and preprocess a single image"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    return image


def create_test_dataset(data_dir, batch_size=8, image_size=(512, 512)):
    """Create test dataset from folder structure"""
    image_paths = []
    labels = []

    # Process original images (label = 0)
    original_dir = os.path.join(data_dir, "original")
    if os.path.exists(original_dir):
        for image_file in os.listdir(original_dir):
            if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(original_dir, image_file))
                labels.append(0)

    # Process duplicate images (label = 1)
    duplicate_dir = os.path.join(data_dir, "duplicate")
    if os.path.exists(duplicate_dir):
        for image_file in os.listdir(duplicate_dir):
            if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(duplicate_dir, image_file))
                labels.append(1)

    print(
        f"Test set - Original: {len([l for l in labels if l == 0])}, Duplicate: {len([l for l in labels if l == 1])}"
    )

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def preprocess_fn(image_path, label):
        image = load_image(image_path, image_size)
        return image, label

    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, image_paths, labels


def load_trained_model(weights_path, tile_size=256, tile_stride=192, num_classes=2):
    """Load the trained model"""
    model = E2EDetectionModel(
        tile_size=tile_size, tile_stride=tile_stride, num_classes=num_classes
    )

    # Build the model by calling it with dummy input
    dummy_input = tf.random.normal((1, 512, 512, 3))
    _ = model(dummy_input)

    # Load weights
    model.load_weights(weights_path)
    print(f"Model loaded from {weights_path}")

    return model


def evaluate_model(model, test_dataset, class_names=["Original", "Duplicate"]):
    """Evaluate model on test dataset"""
    print("Evaluating model...")

    # Get predictions
    all_predictions = []
    all_labels = []

    for batch_images, batch_labels in test_dataset:
        predictions = model(batch_images, training=False)
        predicted_classes = tf.argmax(predictions, axis=1)

        all_predictions.extend(predicted_classes.numpy())
        all_labels.extend(batch_labels.numpy())

    # Calculate metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"Test Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    return accuracy, all_predictions, all_labels


def test_single_image(model, image_path, class_names=["Original", "Duplicate"]):
    """Test a single image"""
    print(f"\nTesting single image: {image_path}")

    # Load and preprocess image
    image = load_image(image_path)
    image_batch = tf.expand_dims(image, 0)  # Add batch dimension

    # Get prediction
    prediction = model(image_batch, training=False)
    probabilities = tf.nn.softmax(prediction, axis=1)
    predicted_class = tf.argmax(prediction, axis=1)[0].numpy()
    confidence = tf.reduce_max(probabilities, axis=1)[0].numpy()

    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    print(
        f"Probabilities - Original: {probabilities[0][0]:.4f}, Duplicate: {probabilities[0][1]:.4f}"
    )

    # Display image
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(
        f"Prediction: {class_names[predicted_class]} (Confidence: {confidence:.4f})"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return predicted_class, confidence


def test_image_folder(model, folder_path, class_names=["Original", "Duplicate"]):
    """Test all images in a folder"""
    print(f"\nTesting images in folder: {folder_path}")

    results = []
    image_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            predicted_class, confidence = test_single_image(
                model, image_path, class_names
            )
            results.append(
                {
                    "filename": image_file,
                    "predicted_class": class_names[predicted_class],
                    "confidence": confidence,
                }
            )
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Summary
    print(f"\nSummary for {len(results)} images:")
    for result in results:
        print(
            f"{result['filename']}: {result['predicted_class']} ({result['confidence']:.4f})"
        )

    return results


def compare_images(
    model, image_path1, image_path2, class_names=["Original", "Duplicate"]
):
    """Compare two images side by side"""
    print(f"\nComparing images:")
    print(f"Image 1: {image_path1}")
    print(f"Image 2: {image_path2}")

    # Load images
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)

    # Get predictions
    pred1_class, conf1 = test_single_image(model, image_path1, class_names)
    pred2_class, conf2 = test_single_image(model, image_path2, class_names)

    # Display comparison
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title(f"Image 1\n{class_names[pred1_class]} ({conf1:.4f})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title(f"Image 2\n{class_names[pred2_class]} ({conf2:.4f})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """Main testing function"""
    # Configuration
    WEIGHTS_PATH = "final_model.weights.h5"  # or "best_model.h5"
    TEST_DIR = r"D:\newresearche2e\my_custom_coco_data\test"

    # Load model
    try:
        model = load_trained_model(WEIGHTS_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Test on test dataset
    if os.path.exists(TEST_DIR):
        test_dataset, image_paths, labels = create_test_dataset(TEST_DIR)
        accuracy, predictions, true_labels = evaluate_model(model, test_dataset)
    else:
        print(f"Test directory {TEST_DIR} not found")

    # Interactive testing
    while True:
        print("\n" + "=" * 50)
        print("Testing Options:")
        print("1. Test single image")
        print("2. Test all images in folder")
        print("3. Compare two images")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                test_single_image(model, image_path)
            else:
                print("Image not found!")

        elif choice == "2":
            folder_path = input("Enter folder path: ").strip()
            if os.path.exists(folder_path):
                test_image_folder(model, folder_path)
            else:
                print("Folder not found!")

        elif choice == "3":
            image1_path = input("Enter first image path: ").strip()
            image2_path = input("Enter second image path: ").strip()
            if os.path.exists(image1_path) and os.path.exists(image2_path):
                compare_images(model, image1_path, image2_path)
            else:
                print("One or both images not found!")

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
