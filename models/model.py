import torch.nn as nn
import torch
import pytorch_lightning as pl
import torchmetrics
from tml.model.dropout import CustomDropout
import numpy as np


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim, nb_classes, dropout_rate_train=0.0, dropout_rate_pred=0.8):
        """
        Binary Classification Model with configurable dropout.

        Args:
            input_dim (int): Number of input features.
            nb_classes (int): Number of output classes.
            dropout_rate_train (float): Dropout rate during training.
            dropout_rate_pred (float): Dropout rate during prediction.
        """
        super(BinaryClassificationModel, self).__init__()

        # self.custom_dropout1 = CustomDropout(p_train=dropout_rate_train, p_pred=0.8)
        # self.custom_dropout2 = CustomDropout(p_train=dropout_rate_train, p_pred=0.7)
        

        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            # self.custom_dropout1,  # First Dropout layer
            nn.Dropout(0.7, ),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            # self.custom_dropout2,  # Second Dropout layer
            nn.Dropout(0.8),
            nn.Linear(input_dim // 2, nb_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    # def enable_dropout(self):
    #     """Enable dropout layers for prediction with dropout."""
    #     self.custom_dropout1.enable_dropout()
    #     self.custom_dropout2.enable_dropout()

    # def disable_dropout(self):
    #     """Disable dropout layers."""
    #     self.custom_dropout1.disable_dropout()
    #     self.custom_dropout2.disable_dropout()

class BinaryClassificationLightning(pl.LightningModule):
    def __init__(self, input_dim, nb_classes=1, dropout_rate_train=0.0, dropout_rate_pred=0.8, learning_rate=1e-3):
        """
        PyTorch Lightning Module for Binary Classification.

        Args:
            input_dim (int): Number of input features.
            nb_classes (int, optional): Number of output classes. Defaults to 1.
            dropout_rate_train (float, optional): Dropout rate during training. Defaults to 0.0.
            dropout_rate_pred (float, optional): Dropout rate during prediction. Defaults to 0.8.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        """
        super(BinaryClassificationLightning, self).__init__()
        self.learning_rate = learning_rate
        self.model = BinaryClassificationModel(
            input_dim, nb_classes, 
            dropout_rate_train=dropout_rate_train, 
            dropout_rate_pred=dropout_rate_pred
        )
        self.loss_fn = nn.BCELoss()  # Use BCELoss since model outputs probabilities
        self.train_accuracy = torchmetrics.Accuracy(task='binary', threshold=0.5)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y.float())
        acc = self.train_accuracy(y_pred, y.int())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict(self, dataloader):
        """
        Predict without dropout, using a DataLoader.
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader containing input data.
        Returns:
            numpy.ndarray: Predicted probabilities.
        """
        self.eval()
        # self.model.disable_dropout()  # Ensure dropout is disabled
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                X_batch = batch[0] if isinstance(batch, (list, tuple)) else batch  # Handle case where batch is (X, y)
                y_pred_batch = self(X_batch)
                all_preds.append(y_pred_batch.cpu().numpy())

        return np.concatenate(all_preds, axis=0)  # Concatenate all batch predictions


    # def predict_with_dropout(self, dataloader, n_iter=10):
    #     """
    #     Predict with dropout active for uncertainty estimation (MC Dropout).
    #     Args:
    #         dataloader (torch.utils.data.DataLoader): DataLoader containing input data.
    #         n_iter (int, optional): Number of stochastic forward passes. Defaults to 10.
    #     Returns:
    #         numpy.ndarray: Predicted probabilities from multiple forward passes, with shape [n_iter, total_samples].
    #     """
    #     self.eval()  # Ensure layers like BatchNorm are in eval mode
    #     self.model.enable_dropout()  # Activate dropout layers
    #     all_preds = []

    #     # Perform multiple stochastic forward passes
    #     for _ in range(n_iter):
    #         batch_preds = []
    #         with torch.no_grad():
    #             for batch in dataloader:
    #                 X_batch = batch[0] if isinstance(batch, (list, tuple)) else batch  # Handle case where batch is (X, y)
    #                 X_batch = X_batch.float()  # Ensure input is float32
    #                 y_pred = self(X_batch)
    #                 batch_preds.append(y_pred.cpu())  # Move to CPU to avoid memory issues

    #         # Concatenate all batch predictions for this iteration
    #         all_preds.append(torch.cat(batch_preds, dim=0).unsqueeze(0))  # Shape: [1, total_samples]

    #     # Concatenate predictions from all iterations
    #     all_preds = torch.cat(all_preds, dim=0)  # Shape: [n_iter, total_samples]
    #     self.model.disable_dropout()  # Disable dropout after prediction
    #     return all_preds.numpy()
