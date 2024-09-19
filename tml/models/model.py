import torch.nn as nn
import torch

def build_model(input_dim, nb_classes, model_type='ml-binary'):
    if model_type == 'ml-binary':
        return BinaryClassificationModel(input_dim, nb_classes)
    elif model_type == 'ml-binary-dropout':
        return BinaryClassificationModel(input_dim, nb_classes, dropout=True)

class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim, nb_classes, dropout=False):
        super(BinaryClassificationModel, self).__init__()
        layers = [
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, nb_classes),
            nn.Sigmoid()
        ]
        if dropout:
            layers.insert(2, nn.Dropout(0.5))  # Add dropout layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
