import pytorch_lightning as pl
import timm
import torch
from src.data_preprocessing import create_data_loader
from src.model import init_model

class CoatTimmModel(pl.LightningModule):
    def __init__(self, model_name='coatnet_rmlp_2_rw_224', num_classes=8, learning_rate=3.9810717055349735e-05):
        super().__init__()
        self.model = init_model(model_name, num_classes)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.model_name = model_name


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
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
