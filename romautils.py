from pyexpat import model
from torch import nn
import aiogram
from io import BytesIO
from PIL import Image
import pytorch_lightning as lg
import torch
import torchvision.transforms as T
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import streamlit as st


class Model(nn.Module):
    def __init__(self,enet):
        super().__init__()
        self.model = enet
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model._dropout = nn.Dropout(0.0)
        self.model._fc = nn.Sequential(
            nn.Linear(self.model._fc.in_features, 1000),
            nn.LeakyReLU(),
            nn.Dropout(0.13),
            nn.Linear(1000, 4),
            
        )
        ##nn.init.kaiming_normal_(list(self.model._fc.parameters()), mode='fan_out', nonlinearity='leaky_relu')
    
    def forward(self, x):
        x = self.model(x)
        return x

class MyModel(lg.LightningModule):  
    def __init__(self):
        super().__init__()
        self.model = Model(EfficientNet.from_pretrained('efficientnet-b7'))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        self.early_stopping = lg.callbacks.EarlyStopping(
            monitor='val_acc',  
            min_delta=0.004,      
            patience=7,         
            verbose=True,
            mode='max'           
        )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        accuracy = (torch.argmax(y_pred, dim=1) == y).float().mean()
        f1loss=multiclass_f1_score(y_pred, y, num_classes=4)
        self.log('train_acc', accuracy)
        self.log('train_loss', loss)
        self.log('train_f1', f1loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        accuracy = (torch.argmax(y_pred, dim=1) == y).float().mean()
        f1loss= multiclass_f1_score(y_pred, y, num_classes=4)
        self.log('val_acc', accuracy)
        self.log('val_loss', loss)
        self.log('val_f1', f1loss)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer
    
    def on_epoch_end(self, epoch, logs=None):
        self.scheduler.step()
        if epoch % 10 == 0:
            self.early_stopping.step(logs['val_acc'])
            if self.early_stopping.early_stop:
                print("Early stopping")
                self.trainer.save_checkpoint("best_model.pt")
                self.trainer.save_checkpoint("last_model.pt")
                return


def get_model():
    model=torch.load("bruhmodel62.pt",map_location=torch.device('cpu'))
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