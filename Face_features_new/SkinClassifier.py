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
ImageFile.LOAD_TRUNCATED_IMAGES = True
plt.ion()


batch_size = 5
model_path = "C:/Users/thend/Desktop/Pratik/Face_features/Models/"
model_name = "skinclassifier_oncpu.pt"
model_path = model_path + model_name
num_workers = 0


transform = transforms.Compose([
    transforms.Resize(size=(256,256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_set = './Face_features/Dataset/Training'
valid_set = './Face_features/Dataset/Validation'

train_data = datasets.ImageFolder(train_set, transform=transform)
valid_data = datasets.ImageFolder(valid_set, transform=transform)

a = train_data.class_to_idx
print(a)
# print(a["Train"])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

loaders = {
    'train': train_loader,
    'valid': valid_loader
}

class_names = ['Acne', 'Pale_skintone', 'Pigmentation', 'Pore_Quality', 'Wrinkled', 'dark_skintone', 'light_skintone', 'medium_skintone']

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
    plt.pause(0.001)  # pause a bit so that plots are updated

class_names = loaders["train"].dataset.classes
print(class_names)

# Get a batch of training data
inputs, classes = next(iter(train_loader))
# print(inputs, classes)

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Sequential(torch.nn.Linear(2048,128),
                                      torch.nn.ReLU(),
                                       torch.nn.Linear(128,8),
                                       torch.nn.Softmax()
                                      )

model_transfer = model

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_transfer.fc.parameters(), lr=0.0005)

n_epochs = 10

train_accuracy_list = []
train_loss_list = []
valid_accuracy_list = []
valid_loss_list = []

def train(n_epochs, loader, model, optimizer, criterion, save_path):
    
    valid_loss_min = np.Inf
       
    for epoch in range(1, (n_epochs+1)):
        
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0
        
        model.train()
        
        for batch_idx, (data, target) in enumerate(loader['train']):
            

            optimizer.zero_grad()
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_acc = train_acc + torch.sum(preds == target.data)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            model.eval()
        for batch_idx, (data, target) in enumerate(loader['valid']):

            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            
            valid_acc = valid_acc + torch.sum(preds == target.data)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
        train_loss = train_loss/len(loader['train'].dataset)
        valid_loss = valid_loss/len(loader['valid'].dataset)
        train_acc = train_acc/len(loader['train'].dataset)
        valid_acc = valid_acc/len(loader['valid'].dataset)
        
        train_accuracy_list.append(train_acc)
        train_loss_list.append(train_loss)
        valid_accuracy_list.append(valid_acc)
        valid_loss_list.append(valid_loss)
        
        print('Epoch: {} \tTraining Acc: {:6f} \tTraining Loss: {:6f} \tValidation Acc: {:6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_acc,
            train_loss,
            valid_acc,
            valid_loss
            ))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss  
            
    return model

print(model_path)
model = train(n_epochs, loaders, model, optimizer, criterion, model_path)

plt.style.use("ggplot")
plt.figure()
loss_lis = [tensor.item() for tensor in train_loss_list]

""" The training loss should decrease over time as the model learns from the data. 
However, a very low training loss doesn't necessarily mean the model 
will perform well on new, unseen data, as it may have overfit the training data"""

print("Last loss: ", train_loss_list[-1])
plt.plot(loss_lis, label="train_loss")
plt.title("Train-Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")

plt.style.use("ggplot")
plt.figure()

print()
train_acc_lis = [tensor.item() for tensor in train_accuracy_list]
valid_acc_lis = [tensor.item() for tensor in valid_accuracy_list]

mean_accuracy_train = torch.tensor(train_accuracy_list)
mean_accuracy_train = mean_accuracy_train.mean()
print("Train accuracy mean: ", mean_accuracy_train)
mean_accuracy_valid = torch.tensor(valid_accuracy_list)
mean_accuracy_valid = mean_accuracy_valid.mean()
print("Valid accuracy mean: ", mean_accuracy_valid)

plt.plot(train_acc_lis, label="train_acc")
plt.plot(valid_acc_lis, label="valid_acc")

plt.title("Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")


from PIL import Image

class_names = ['Acne', 'Pale_skintone', 'Pigmentation', 'Pore_Quality', 'Wrinkled', 'dark_skintone', 'light_skintone', 'medium_skintone']

def predict(image, model_path):
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    

    model.load_state_dict(torch.load(model_path))
    # model.to(device=device)
    # model.to("cpu")
    model.eval()

    pred = model(image)
    idx = torch.argmax(pred)
    print(idx, "idx")
    prob = pred[0][idx].item()*100

    # print(class_names[idx], "class_names[idx]")
    
    return class_names[idx], prob



def test(image_path, model_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()

    prediction, prob = predict(img, model_path=model_path)
    print(prediction, prob)


test(image_path="C:/Users/thend/Downloads/download (2).jpg", model_path="C:/Users/thend/Desktop/Pratik/Face_features/Models/Modelsskinclassifier.pt")



import torch
from torch import nn
from torchvision import transforms
from PIL import Image

def load_model(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(model_path, image_path, prediction_transform):
    # Load the model
    model = load_model(model_path)

    # Read and transform the image
    image = Image.open(image_path).convert("RGB")
    transformed_image = prediction_transform(image)
    transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(transformed_image)

    # Get predicted class
    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

# Example usage:
model_path = './Face_features/Models/Modelsskinclassifier.pt'
image_path = 'image to predict'
prediction_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

predicted_class = predict_image(model_path, image_path, prediction_transform)
print("Predicted Class:", predicted_class)

# Load the model
model.load_state_dict(torch.load('C:/Users/thend/Desktop/Pratik/Face_features/Models/Modelsskinclassifier.pt', map_location=torch.device('cpu')))
model.eval()

# Provide an example input
example_input = torch.randn(5, 3, 224, 224)

# Export the model to ONNX
torch.onnx.dynamo_export(model, example_input)





