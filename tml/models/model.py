import torch.nn as nn
import torch
import pytorch_lightning as pl
import torchmetrics



# def build_model(input_dim, nb_classes, model_type='ml-binary', dropout_rate=0.5):
#     if model_type == 'ml-binary':
#         return BinaryClassificationModel(input_dim, nb_classes)
#     elif model_type == 'ml-binary-dropout':
#         return BinaryClassificationModel(input_dim, nb_classes, dropout=True)

class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim, nb_classes, dropout_rate=None):
        super(BinaryClassificationModel, self).__init__()

        layers = [
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, int(input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim/2), nb_classes),
            nn.Sigmoid()
        ]
        if dropout_rate:
            layers.insert(2, nn.Dropout(dropout_rate))  # Add dropout layer
            layers.insert(2, nn.Dropout(dropout_rate))  # Add dropout layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Define the LightningModule with prediction methods
class BinaryClassificationLightning(pl.LightningModule):
    def __init__(self, input_dim, nb_classes=2, dropout_rate=None, learning_rate=1e-3):
        super(BinaryClassificationLightning, self).__init__()
        self.learning_rate = learning_rate
        self.model = BinaryClassificationModel(input_dim, nb_classes, dropout_rate)
        self.loss_fn = nn.BCELoss()  # Use BCELoss since model outputs probabilities
        self.train_accuracy = torchmetrics.Accuracy(threshold=0.5)

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

    # Add the predict method to the model class
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            y_pred = self(X_tensor)
        return y_pred.numpy()

    # Add the predict_with_dropout method to the model class
    def predict_with_dropout(self, X, n_iter):
        self.train()  # Enable dropout during inference
        preds = []
        X_tensor = torch.from_numpy(X).float()
        for _ in range(n_iter):
            with torch.no_grad():
                y_pred = self(X_tensor)
                preds.append(y_pred.unsqueeze(0))  # Shape: [1, batch_size]
        preds = torch.cat(preds, dim=0)  # Shape: [n_iter, batch_size]
        return preds.numpy()
