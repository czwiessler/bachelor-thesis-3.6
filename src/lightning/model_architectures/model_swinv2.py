import pytorch_lightning as pl
import timm
import torch
from src.data_preprocessing import create_data_loader
from src.model import init_model
from transformers import AutoImageProcessor, AutoModelForImageClassification

class Swinv2TimmModel(pl.LightningModule):
    def __init__(self, model_name='microsoft/swinv2-base-patch4-window12-192-22k', num_classes=8, learning_rate=3.9810717055349735e-05):
        super().__init__()
        self.model = init_model(model_name, num_classes)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)


    def forward(self, x):
        return self.model(x)

    def process_inputs(self, inputs):
        # Process images and extract pixel values
        processed_inputs = self.processor(images=inputs, return_tensors='pt')
        return processed_inputs['pixel_values']

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = self.process_inputs(inputs)
        outputs = self(inputs)
        logits = outputs.logits
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = self.process_inputs(inputs)
        outputs = self(inputs)
        logits = outputs.logits
        loss = self.criterion(logits, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        train_loader, _ = create_data_loader(model_name=self.model_name)
        return train_loader

    def val_dataloader(self):
        _, val_loader = create_data_loader(model_name=self.model_name)
        return val_loader
