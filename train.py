#!/usr/bin/env python3

import argparse
import platform
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# Hyperparameters and model directory
BATCH_SIZE = 64
EPOCHS = 2
LR = 0.01
MODEL_DIR = os.getenv("MODEL_DIR", "/mnt/models")  # Default location for saving the model

# Define the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Fully connected layers
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # First conv layer with max pooling and ReLU activation
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Second conv layer with max pooling and ReLU activation
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # Flatten the tensor
        x = x.view(-1, 320)
        # Fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        # Final layer outputs raw scores for each of the 10 classes
        x = self.fc2(x)
        return x

def train(device, rank=0):
    print("Using device:", device)
    # Define the transformation pipeline for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST training dataset
    train_dataset = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create model and move it to the chosen device
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    model.train()  # Set the model to training mode
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()         # Clear gradients from the previous step
            output = model(data)          # Forward pass: compute predictions
            loss = F.cross_entropy(output, target)  # Compute cross-entropy loss
            loss.backward()               # Backpropagation: compute gradients
            optimizer.step()              # Update model weights

            # Log training progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")

    # Save the trained model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "mnist_cnn.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif platform.system().lower() == "darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train(device)
