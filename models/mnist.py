import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics.classification import Accuracy

class CNNBinaryMNISTClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, dropout_rate=0.5):
        super(CNNBinaryMNISTClassifier, self).__init__()
        self.learning_rate = learning_rate
        
        # Define the CNN architecture using nn.Sequential
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),  # (28, 28) -> (28, 28)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # (28, 28) -> (28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (28, 28) -> (14, 14)
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # (14, 14) -> (14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (14, 14) -> (14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (14, 14) -> (7, 7)
            
            nn.Flatten(),  # Flatten the output for the fully connected layers
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),  # Single output for binary classification
            nn.Sigmoid()  # Apply sigmoid to get output between 0 and 1
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().unsqueeze(1)  # Adjust shape for BCELoss
        logits = self(x)
        loss = nn.BCELoss()(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().unsqueeze(1)  # Adjust shape for BCELoss
        logits = self(x)
        loss = nn.BCELoss()(logits, y)
        
        # Calculate accuracy
        preds = (logits > 0.5).float()  # Threshold logits at 0.5 to make binary predictions
        accuracy = (preds == y).float().mean()

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)
        return {"test_loss": loss, "test_accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
