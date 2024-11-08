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
        if not torch.equal(unique_classes, torch.tensor([0, 1])):
            raise ValueError("The dataset should contain labels 0 and 1.")

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

class BalancedSampler:
    def __init__(self, dataset, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # labels should be 0 or 1
        labels = dataset.labels

        # # Separate indices for each class based on identified labels
        # unique_classes = torch.unique(labels)
        # self.class_0_indices = (labels == unique_classes[0]).nonzero(as_tuple=True)[0]
        # self.class_1_indices = (labels == unique_classes[1]).nonzero(as_tuple=True)[0]

        self.class_0_indices = (labels == 0).nonzero(as_tuple=True)[0]
        self.class_1_indices = (labels == 1).nonzero(as_tuple=True)[0]

        # Determine which is the smaller class
        if len(self.class_0_indices) < len(self.class_1_indices):
            self.smaller_class_indices = self.class_0_indices
            self.larger_class_indices = self.class_1_indices
        else:
            self.smaller_class_indices = self.class_1_indices
            self.larger_class_indices = self.class_0_indices

        self.min_class_size = len(self.smaller_class_indices)
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)  # Use a random generator for efficiency

    def sample_balanced_subset(self):
        sampled_smaller_class_indices = self.smaller_class_indices
        
        # Sample min_class_size from the larger class
        sampled_larger_class_indices = torch.tensor(
            self.rng.choice(self.larger_class_indices.numpy(), self.min_class_size, replace=False),
            dtype=torch.long
        )
        
        sampled_indices = torch.cat((sampled_smaller_class_indices, sampled_larger_class_indices))
        
        sampled_data = self.dataset.data[sampled_indices]
        sampled_labels = self.dataset.labels[sampled_indices]
        
        balanced_dataset = TMLDataset(sampled_data, sampled_labels)
        return balanced_dataset

