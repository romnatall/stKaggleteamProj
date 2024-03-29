from torchvision.models import resnet18, ResNet18_Weights
from turtle import mode
from torch import nn
import aiogram
from io import BytesIO
from PIL import Image
import pytorch_lightning as lg
import torch
import torchvision.transforms as T
from torchvision import transforms

class MyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        #замена слоя
        self.model.fc = nn.Linear(512, 100)
        #разморозка
        for i in self.model.parameters():
            i.requires_grad = False

        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)



def get_model():
    model=torch.load("model.pt")
    model.eval()
    return model

def classify_image(image_path):
    model = get_model()
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities[predicted_class].item()