import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score

def compute_roc_auc(y_true, y_pred):
    """
    Compute ROC curve and AUC score.
    
    Parameters:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted probabilities or decision function.

    Returns:
        fpr (array-like): False positive rates.
        tpr (array-like): True positive rates.
        auc_score (float): Area under the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def compute_average_precision(y_true, y_pred):
    """
    Compute average precision (AP) score.
    
    Parameters:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted probabilities or decision function.

    Returns:
        ap_score (float): Average precision score.
    """
    ap_score = average_precision_score(y_true, y_pred)
    return ap_score


def compute_tpr_cutoff(fpr, tpr, target_tpr=0.95):
    """
    Compute the cutoff threshold to achieve a specific True Positive Rate (TPR).
    
    Parameters:
        fpr (array-like): False positive rates from ROC curve.
        tpr (array-like): True positive rates from ROC curve.
        target_tpr (float): Desired TPR threshold. Default is 0.95.

    Returns:
        cutoff (float): Cutoff threshold for achieving the target TPR.
    """
    cutoff_idx = np.argmax(tpr >= target_tpr)
    return fpr[cutoff_idx], tpr[cutoff_idx]


def compute_classification_threshold(y_true, y_pred, auc_cutoff=0.9, pscore_cutoff=0.2):
    """
    Compute final classification threshold based on AUC and probability score cutoffs.
    
    Parameters:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted probabilities or decision function.
        auc_cutoff (float): AUC threshold to classify as a valid detection.
        pscore_cutoff (float): Probability score cutoff.

    Returns:
        result (str): Classification result based on the threshold.
    """
    fpr, tpr, auc_score = compute_roc_auc(y_true, y_pred)
    if auc_score >= auc_cutoff and np.mean(y_pred) > pscore_cutoff:
        return "PASS"
    return "FAIL"

