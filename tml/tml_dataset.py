import numpy as np
import torch
from torch.utils.data import Dataset

class TMLDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # Assume data is already a torch.Tensor
        self.labels = labels  # Assume labels is already a torch.Tensor
        

        unique_classes = torch.unique(self.labels)
        if len(unique_classes) != 2:
            raise ValueError("The dataset should contain exactly two unique classes.")
        if not torch.equal(unique_classes, torch.tensor([0, 1], device=self.data.device)):
            raise ValueError("The dataset should contain labels 0 and 1.")

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

import numpy as np
import torch
from torch.utils.data import Dataset

class BalancedSampler:
    def __init__(self, dataset, indices=None, seed=42):
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Use subset of dataset if indices are provided
        if indices is not None:
            # Filter data and labels based on the provided indices
            data = dataset.data[indices]
            labels = dataset.labels[indices]
            self.dataset = TMLDataset(data, labels)
        else:
            # Use the original dataset
            self.dataset = dataset
        
        # Assume labels should be binary (0 and 1)
        labels = self.dataset.labels
        self.rng = np.random.default_rng(seed)

        # Separate indices by class
        self.class_0_indices = (labels == 0).nonzero(as_tuple=True)[0]
        self.class_1_indices = (labels == 1).nonzero(as_tuple=True)[0]

        # Determine smaller and larger class indices
        if len(self.class_0_indices) <= len(self.class_1_indices):
            self.smaller_class_indices, self.larger_class_indices = self.class_0_indices, self.class_1_indices
        else:
            self.smaller_class_indices, self.larger_class_indices = self.class_1_indices, self.class_0_indices

        self.min_class_size = len(self.smaller_class_indices)

    def sample_balanced_subset(self):
        # Sample all from smaller class and an equal number from the larger class
        sampled_larger_class_indices = torch.tensor(
            # self.rng.choice(self.larger_class_indices.numpy(), self.min_class_size, replace=False),
            self.rng.choice(self.larger_class_indices.cpu().numpy(), self.min_class_size, replace=False),
            dtype=torch.long
        )
        # sampled_indices = torch.cat((self.smaller_class_indices, sampled_larger_class_indices))
        sampled_indices = torch.cat((self.smaller_class_indices.to(sampled_larger_class_indices.device), sampled_larger_class_indices))

        
        # Create the balanced dataset from sampled indices
        sampled_data = self.dataset.data[sampled_indices]
        sampled_labels = self.dataset.labels[sampled_indices]
        return TMLDataset(sampled_data, sampled_labels)


# Function to prune the test set
def prune(labels, y_pred, lower_thresh=0.3, upper_thresh=0.7):
    """
    Prune the test set based on the predicted probabilities and thresholds.
    """
    TPs = [ind for ind, (label, pred) in enumerate(zip(labels, y_pred)) if label == 1 and pred > lower_thresh]
    TNs = [ind for ind, (label, pred) in enumerate(zip(labels, y_pred)) if label == 0 and pred < upper_thresh]
    return TPs, TNs