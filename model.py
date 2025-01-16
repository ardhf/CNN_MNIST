import torch.nn as nn

class PureCNNModel(nn.Module):
    def __init__(self):
        super(PureCNNModel, self).__init__()
        self.cnn_layers = nn.Sequential(
            # 28x28x1 grayscale input image
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # 28x28x32 beacuse the 32 output layers
            nn.MaxPool2d(kernel_size=2, stride=2),  # size gets cut in half because of the 2 kernel size and 2 stride

            nn.Dropout2d(0.1),  # Each neuron in the layer has a 10% chance of being zeroed out

            # 14x14x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # 14x14x64
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(0.1),  # Each neuron in the layer has a 10% chance of being zeroed out

            # 7x7x64
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # 7x7x32
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3x3x32
            nn.Flatten(),
            # 1x1x288
            nn.Linear(288, 10)  # Final classification layer (10 classes)
        )
        

    def forward(self, x):
        x = self.cnn_layers(x)  # Apply convolutional layers
        return x
