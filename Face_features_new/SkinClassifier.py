# Import necessary libraries
import os
import torch
from torchvision import datasets
import torchvision
from torchvision.transforms import transforms
import torchvision.models as models
import numpy as np
from PIL import ImageFile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
plt.ion()  # Interactive mode on for matplotlib

# Define batch size and paths
batch_size = 5
model_path = "./Face_features/Models/"
model_name = "skinclassifier_oncpu.pt"
model_path = model_path + model_name
num_workers = 0

# Define image transformations for training and validation datasets
transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define dataset paths
train_set = './Face_features/Dataset/Training'
valid_set = './Face_features/Dataset/Validation'

# Load datasets with defined transformations
train_data = datasets.ImageFolder(train_set, transform=transform)
valid_data = datasets.ImageFolder(valid_set, transform=transform)

# Print class-to-index mapping
a = train_data.class_to_idx
print(a)

# Create data loaders for training and validation datasets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

# Store loaders in a dictionary for easy access
loaders = {
    'train': train_loader,
    'valid': valid_loader
}

# Define class names for the dataset
class_names = ['Acne', 'Pale_skintone', 'Pigmentation', 'Pore_Quality', 'Wrinkled', 'dark_skintone', 'light_skintone', 'medium_skintone']

# Function to display an image tensor
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pause to update plots

# Get class names from the dataset
class_names = loaders["train"].dataset.classes
print(class_names)

# Get a batch of training data
inputs, classes = next(iter(train_loader))

# Make a grid from batch and display it
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

# Load a pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)

# Freeze model parameters to avoid updating during training
for param in model.parameters():
    param.requires_grad = False

# Replace the fully connected layer to match the number of classes
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 8),
    torch.nn.Softmax()
)

# Use the modified model
model_transfer = model

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_transfer.fc.parameters(), lr=0.0005)

# Number of training epochs
n_epochs = 10

# Lists to store training and validation metrics
train_accuracy_list = []
train_loss_list = []
valid_accuracy_list = []
valid_loss_list = []

# Training function
def train(n_epochs, loader, model, optimizer, criterion, save_path):
    valid_loss_min = np.Inf  # Track change in validation loss
    
    for epoch in range(1, (n_epochs+1)):
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0
        
        model.train()  # Set model to training mode
        
        for batch_idx, (data, target) in enumerate(loader['train']):
            optimizer.zero_grad()  # Clear previous gradients
            output = model(data)  # Forward pass
            _, preds = torch.max(output, 1)  # Get predictions
            loss = criterion(output, target)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            train_acc += torch.sum(preds == target.data)  # Calculate accuracy
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))  # Update loss
            
        model.eval()  # Set model to evaluation mode
        for batch_idx, (data, target) in enumerate(loader['valid']):
            output = model(data)  # Forward pass
            _, preds = torch.max(output, 1)  # Get predictions
            loss = criterion(output, target)  # Calculate loss
            
            valid_acc += torch.sum(preds == target.data)  # Calculate accuracy
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))  # Update loss
            
        train_loss /= len(loader['train'].dataset)
        valid_loss /= len(loader['valid'].dataset)
        train_acc /= len(loader['train'].dataset)
        valid_acc /= len(loader['valid'].dataset)
        
        train_accuracy_list.append(train_acc)
        train_loss_list.append(train_loss)
        valid_accuracy_list.append(valid_acc)
        valid_loss_list.append(valid_loss)
        
        print(f'Epoch: {epoch} \tTraining Acc: {train_acc:.6f} \tTraining Loss: {train_loss:.6f} \tValidation Acc: {valid_acc:.6f} \tValidation Loss: {valid_loss:.6f}')

        if valid_loss <= valid_loss_min:
            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model ...')
            torch.save(model.state_dict(), save_path)  # Save model if validation loss decreases
            valid_loss_min = valid_loss  
            
    return model

# Print model path
print(model_path)
# Train the model
model = train(n_epochs, loaders, model, optimizer, criterion, model_path)

# Plot training loss
plt.style.use("ggplot")
plt.figure()
loss_lis = [tensor.item() for tensor in train_loss_list]

print("Last loss: ", train_loss_list[-1])
plt.plot(loss_lis, label="train_loss")
plt.title("Train-Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")

# Plot training and validation accuracy
plt.style.use("ggplot")
plt.figure()

train_acc_lis = [tensor.item() for tensor in train_accuracy_list]
valid_acc_lis = [tensor.item() for tensor in valid_accuracy_list]

mean_accuracy_train = torch.tensor(train_accuracy_list).mean()
print("Train accuracy mean: ", mean_accuracy_train)
mean_accuracy_valid = torch.tensor(valid_accuracy_list).mean()
print("Valid accuracy mean: ", mean_accuracy_valid)

plt.plot(train_acc_lis, label="train_acc")
plt.plot(valid_acc_lis, label="valid_acc")

plt.title("Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

# Prediction function
def predict(image, model_path):
    prediction_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = prediction_transform(image)[:3, :, :].unsqueeze(0)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    pred = model(image)
    idx = torch.argmax(pred)
    print(idx, "idx")
    prob = pred[0][idx].item() * 100
    
    return class_names[idx], prob

# Function to test the prediction function
def test(image_path, model_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()

    prediction, prob = predict(img, model_path=model_path)
    print(prediction, prob)

# Test the model with a sample image
test(image_path="C:/Users/thend/Downloads/download (2).jpg", model_path="./Face_features/Models/Modelsskinclassifier.pt")

# Additional functions for loading and predicting images
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

def load_model(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(model_path, image_path, prediction_transform):
    model = load_model(model_path)
    image = Image.open(image_path).convert("RGB")
    transformed_image = prediction_transform(image)
    transformed_image = transformed_image.unsqueeze(0)

    with torch.no_grad():
        output = model(transformed_image)

    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Example usage of prediction functions
model_path = './Face_features
