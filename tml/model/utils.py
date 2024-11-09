import numpy as np
import torch

def get_input_shape(data):
    """
    Get the input shape for a neural network by removing the batch dimension.
    Handles general cases like tabular, image, and sequence data.
    """

    if isinstance(data, torch.Tensor):
        data_shape = data.size()
    else:
        raise ValueError("Unsupported data type. Expecting torch tensor.")

    input_shape = data_shape[1:]

    return input_shape


def ensure_tensors(data, hard_targets):
    """
    Ensure that `data` and `hard_targets` are tensors.
    """

    # Check if `data` is a tensor; if not, convert it
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

    # Check if `hard_targets` is a tensor; if not, convert it
    if not isinstance(hard_targets, torch.Tensor):
        hard_targets = torch.tensor(hard_targets)

    return data, hard_targets
