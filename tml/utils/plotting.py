import matplotlib.pyplot as plt
import numpy as np

def plot_roc_curve(fpr, tpr, auc_score, save_path=None):
    """
    Plot the ROC curve and save it as an image.
    
    Parameters:
        fpr (array-like): False positive rates.
        tpr (array-like): True positive rates.
        auc_score (float): AUC score for the ROC curve.
        save_path (str): Path to save the plot. If None, the plot is shown interactively.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(precision, recall, ap_score, save_path=None):
    """
    Plot the Precision-Recall curve.
    
    Parameters:
        precision (array-like): Precision values.
        recall (array-like): Recall values.
        ap_score (float): Average precision score.
        save_path (str): Path to save the plot. If None, the plot is shown interactively.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_tml_scores(final_scores, neg_ind, hpos_ind, pscore_cf, auc_cf, tpr_cf, save_path):
    """
    Custom plot for visualizing final TML score results.

    Parameters:
        final_scores (array-like): Final array with predicted scores and uncertainty.
        neg_ind (array-like): Negative sample indices.
        hpos_ind (array-like): Highly positive sample indices.
        pscore_cf (float): Probability score cutoff.
        auc_cf (float): AUC cutoff.
        tpr_cf (float): True Positive Rate cutoff.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 8))

    # Plot score distribution for each type
    plt.scatter(neg_ind, final_scores[neg_ind, 0], color='red', label='Negative Samples')
    plt.scatter(hpos_ind, final_scores[hpos_ind, 0], color='green', label='Highly Positive Samples')

    # Plot probability score cutoff
    plt.axhline(pscore_cf, color='blue', linestyle='--', label=f'PScore Cutoff ({pscore_cf})')

    # Customize plot
    plt.xlabel('Sample Index')
    plt.ylabel('Probability Score')
    plt.title(f'TML Score Results (AUC cutoff: {auc_cf}, TPR cutoff: {tpr_cf})')
    plt.legend()

    # Save plot
    plt.savefig(f"{save_path}_TML_scores.png")
    plt.close()

