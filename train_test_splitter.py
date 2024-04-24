import os
import shutil
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torch

# Define the paths for the source and destination directories
source_dir = "C:/Users/christian.zwiessler/Downloads/train/"
dest_dir = "C:/Users/christian.zwiessler/Downloads/test/"

# Load the dataset from the source directory
dataset = ImageFolder(source_dir)

# Calculate the split size for 10% of the data
test_size = int(0.1 * len(dataset))
train_size = len(dataset) - test_size

# Seed the random generator for reproducibility
torch.manual_seed(42)

# Split the dataset into training and testing
_, test_dataset = random_split(dataset, [train_size, test_size])

# Function to move test images to the destination directory
def move_to_test_set(test_indices):
    # Create subdirectories in the destination directory if they don't exist
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(dest_dir, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)

    # Move each file in the test set
    for idx in test_indices:
        # Get the path of the original file and its destination
        path, class_index = dataset.samples[idx]
        class_name = dataset.classes[class_index]
        file_name = os.path.basename(path)
        destination = os.path.join(dest_dir, class_name, file_name)

        # Move the file
        shutil.move(path, destination)

# Get the indices of the test dataset and move the files
# Get the indices of the test dataset and move the files
test_indices = test_dataset.indices
move_to_test_set(test_indices)

print("Files moved successfully.")

