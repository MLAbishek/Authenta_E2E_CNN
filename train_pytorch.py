import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import os
from src.E2EModel import E2EDetectionModel


def load_image(image_path):
    """Load and preprocess image"""
    image = Image.open(image_path).convert("RGB")
    return image


class ImageDataset(Dataset):
    def __init__(self, data_dir, image_size=(512, 512)):
        """Create dataset from folder structure (original/duplicate folders)"""

        # Get image paths and labels from folder structure
        self.image_paths = []
        self.labels = []

        # Process original images (label = 0)
        original_dir = os.path.join(data_dir, "original")
        if os.path.exists(original_dir):
            for image_file in os.listdir(original_dir):
                if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.image_paths.append(os.path.join(original_dir, image_file))
                    self.labels.append(0)  # 0 for original

        # Process duplicate images (label = 1)
        duplicate_dir = os.path.join(data_dir, "duplicate")
        if os.path.exists(duplicate_dir):
            for image_file in os.listdir(duplicate_dir):
                if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.image_paths.append(os.path.join(duplicate_dir, image_file))
                    self.labels.append(1)  # 1 for duplicate

        print(f"Found {len([l for l in self.labels if l == 0])} original images")
        print(f"Found {len([l for l in self.labels if l == 1])} duplicate images")
        print(f"Total: {len(self.image_paths)} images")

        # Define transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # No normalization here since we want values between 0-1
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = load_image(image_path)
        image = self.transform(image)

        return image, label


def create_dataset(data_dir, batch_size=8, image_size=(512, 512)):
    """Create dataset from folder structure (original/duplicate folders)"""
    dataset = ImageDataset(data_dir, image_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    return dataloader


def train_model(train_dir, val_dir, epochs=10, batch_size=4, learning_rate=0.001):
    """Simple training function"""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataloader = create_dataset(train_dir, batch_size)
    val_dataloader = create_dataset(val_dir, batch_size)

    # Create model
    model = E2EDetectionModel(tile_size=256, tile_stride=128, num_classes=2)
    model.to(device)

    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, verbose=True
    )

    # Training variables
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 5

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)

        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_dataloader)
        train_accuracy = correct_predictions / total_samples

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0

        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_dataloader)
        val_accuracy = val_correct_predictions / val_total_samples

        # Update history
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping and model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    model.load_state_dict(torch.load("best_model.pth"))

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
    print(f"Final validation accuracy: {max(history['val_accuracy']):.4f}")

    # Save final model
    torch.save(model.state_dict(), "final_model.pth")
    print("Model saved as 'final_model.pth'")
