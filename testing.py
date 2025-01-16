import torch
from torchvision import transforms
from PIL import Image
from model import PureCNNModel

handwriting = False

colors = {
    "reset": "\033[0m",  # Reset all attributes
    False: "\033[91m",  # bright red
    True: "\033[92m",  # bright green
}

# Preprocessing function for custom images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure image is grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize as in training
    ])
    image = Image.open(image_path).convert("RGB")  # Open image
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to load model
def load_model(model_path, device):
    model = PureCNNModel()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

# Function to predict the class of an image
def predict_image(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()

if __name__ == "__main__":
    # Load the model
    device = torch.device("cpu")
    model = load_model("mnist_cnn_model.pth", device)

    if handwriting:
        images = ['dj_1', 'dj_2', 'dj_3', 'dj_8']
    else:
        images = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    for image in images:
        image_tensor = preprocess_image(f"images/{image}.jpg")
        predicted_class = predict_image(model, image_tensor, device)
        print(f"{colors[int(image[-1]) == predicted_class]}",
            f"Expected Digit: {image[-1]}, Predicted Digit: {predicted_class}",
            f"{colors['reset']}")
        