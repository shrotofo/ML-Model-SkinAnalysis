# Import necessary libraries
import torch
from torchvision.transforms import transforms
from PIL import Image, ImageFile
import sys

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define the possible class names for predictions
class_names = ['Acne', 'Pale_skintone', 'Pigmentation', 'Pore_Quality', 'Wrinkled', 'dark_skintone', 'light_skintone', 'medium_skintone']

def predict(image, model_path):
    """
    This function takes an image and a model path, applies the necessary transformations to the image,
    loads the model from the given path, and returns the predicted class and its probability.
    """
    # Define the transformations to be applied to the input image
    prediction_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transformations and add a batch dimension
    image = prediction_transform(image)[:3, :, :].unsqueeze(0)

    # Load the model
    model = torch.load(model_path, map_location=torch.device('cpu'))

    # Set the model to evaluation mode
    model.eval()

    # Make the prediction
    pred = model(image)
    idx = torch.argmax(pred)
    prob = pred[0][idx].item() * 100

    return class_names[idx], prob

def test(image_path, model_path):
    """
    This function takes the path to an image and a model, opens the image, and uses the predict function
    to get the prediction and probability.
    """
    # Open the image
    img = Image.open(image_path)

    # Get the prediction and probability
    prediction, prob = predict(img, model_path=model_path)

    return prediction, prob

if __name__ == "__main__":
    # Check if the number of arguments is correct
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_path> <image_path>")
        sys.exit(1)

    # Get the model path and image path from command line arguments
    model_path = sys.argv[1]
    image_path = sys.argv[2]

    # Get the predicted class and probability
    predicted_class, prob = test(image_path, model_path)

    # Print the results
    print("Predicted Class:", predicted_class)
    print("Probability:", prob)
