import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from model import PureCNNModel

num_epochs = 25
load_model = False

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize model
model = PureCNNModel()

# Define the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load model and optimizer if specified
if load_model:
    model.load_state_dict(torch.load("mnist_cnn_model.pth", weights_only=True))
    optimizer.load_state_dict(torch.load("mnist_cnn_model_optimizer.pth", weights_only=True))
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.00001  # Reset to desired initial learning rate
    print("Loaded existing model and optimizer.")

# Initialize learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

start_time = time.time()

# Training loop (for 10 epochs)
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    scheduler.step()
    train_accuracy = 100 * correct_train / total_train

    # Evaluation phase
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test

    # Print metrics for the current epoch
    print(f"Epoch {epoch+1}: Training Loss: {running_loss/len(train_loader):.4f}, "
          f"Training Accuracy: {train_accuracy:.2f}%, Testing Accuracy: {test_accuracy:.2f}% "
          f"Learning Rate: {scheduler.get_last_lr()[0]:.5f}")

# Evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
print(f"total time: {time.time() - start_time} seconds with {device}")

# Save and load the model
torch.save(model.state_dict(), "mnist_cnn_model.pth")
torch.save(optimizer.state_dict(), "mnist_cnn_model_optimizer.pth")