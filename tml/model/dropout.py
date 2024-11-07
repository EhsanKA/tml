import torch.nn as nn


class CustomDropout(nn.Module):
    def __init__(self, p_train=0.0, p_pred=0.8):
        """
        Custom Dropout layer that allows independent control of dropout during training and prediction.

        Args:
            p_train (float): Dropout probability during training.
            p_pred (float): Dropout probability during prediction.
        """
        super(CustomDropout, self).__init__()
        self.p_train = p_train
        self.p_pred = p_pred
        self.dropout_enabled = False  # Flag to control dropout during prediction

    def enable_dropout(self):
        """Enable dropout during prediction."""
        self.dropout_enabled = True

    def disable_dropout(self):
        """Disable dropout during prediction."""
        self.dropout_enabled = False

    def forward(self, x):
        """
        Forward pass for the CustomDropout layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying dropout.
        """
        if self.training:
            p = self.p_train
        elif self.dropout_enabled:
            p = self.p_pred
        else:
            p = 0.0

        if p == 0.0:
            return x
        return nn.functional.dropout(x, p=p, training=True)