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
    n_samples = data_shape[0]

    return input_shape, n_samples


def ensure_tensors(data, hard_targets):
    """
    Ensure that `data` and `hard_targets` are tensors.
    """

    # Check if `data` is a tensor; if not, convert it
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    # Check if `hard_targets` is a tensor; if not, convert it
    if not isinstance(hard_targets, torch.Tensor):
        hard_targets = torch.tensor(hard_targets, dtype=torch.long)

    return data, hard_targets



import numpy as np
from sklearn.metrics import auc
def prepare_data(final, neg_ind, hpos_ind, minScore):
    """Filter and prepare negative and positive sets based on minScore."""
    neg_set = final[neg_ind]
    pos_set = final[hpos_ind] 
    neg_set = neg_set[neg_set[:, 0] > minScore]
    pos_set = pos_set[pos_set[:, 0] > minScore]
    return neg_set, pos_set
def calculate_thresholds_and_tpr(neg_set, pos_set):
    """Calculate thresholds, true positive rates, UTD counts, and false positive rates."""
    thr = []
    tpr = []
    utd = []
    for t in np.arange(0.25, -0.01, -0.01):
        thr.append(t)
        utd_count = len(neg_set[neg_set[:, 1] <= t, 1])
        utd.append(utd_count)
        tpr_value = len(pos_set[pos_set[:, 1] <= t, 1]) / len(pos_set[:, 1])
        tpr.append(tpr_value)
    fpr = np.array(utd) / utd[0] if utd[0] != 0 else np.zeros(len(utd))
    return thr, tpr, utd, fpr
def compute_roc_auc(fpr, tpr):
    """Compute the ROC AUC value."""
    roc_auc = auc(fpr, tpr)
    return roc_auc
def compute_cf(thr, tpr, roc_auc, auc_cf, tpr_cf):
    """Compute the cutoff value cf based on roc_auc and thresholds."""
    cf = 0
    if roc_auc > auc_cf:
        i = 0
        old_thr = 0
        while cf == 0 and i < len(thr):
            if 1 - 4 * thr[i] > tpr[i]:
                cf = (thr[i] + old_thr) / 2
            else:
                old_thr = thr[i]
            i += 1
    else:
        i = len(tpr) - 1
        old_thr = 0
        while cf == 0 and i >= 0:
            if tpr[i] >= tpr_cf:
                cf = (thr[i] + old_thr) / 2
            else:
                old_thr = thr[i]
            i -= 1    
    return cf
def prepare_plot_data(final, neg_ind, hpos_ind):
    """Prepare data by labeling UTDs and Germline SNPs for plotting."""
    data = np.column_stack((final, np.repeat("SNP", len(final))))
    data[neg_ind, 2] = "UTDs"
    data[hpos_ind, 2] = "Germline SNPs"
    return data