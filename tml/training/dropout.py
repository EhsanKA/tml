# import torch

# class DropoutPrediction:
#     def __init__(self, model, n_iter=100):
#         self.model = model
#         self.n_iter = n_iter

#     def predict(self, x):
#         self.model.train()  # Force dropout to stay active
#         predictions = torch.stack([self.model(x) for _ in range(self.n_iter)])
#         return predictions
