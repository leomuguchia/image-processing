import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Load pre-trained VGG16 model
vgg16 = models.vgg16(pretrained=True)

# Remove the classification layers
feature_extractor = nn.Sequential(*list(vgg16.features.children())[:-1])

# Transformation to preprocess the images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_features(image):
    # Preprocess the image
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = feature_extractor(input_tensor)
    return features
