import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc
from tml.plotting.utils import prepare_data, calculate_thresholds_and_tpr, compute_roc_auc, compute_cf, prepare_plot_data

def tml_plots(final, neg_ind, hpos_ind, minScore, auc_cf, tpr_cf, out):

    # Prepare data
    neg_set, pos_set = prepare_data(final, neg_ind, hpos_ind, minScore)
    
    # Calculate thresholds, TPR, UTD, and FPR
    thr, tpr, utd, fpr = calculate_thresholds_and_tpr(neg_set, pos_set)
    
    # Compute ROC AUC
    roc_auc = compute_roc_auc(fpr, tpr)
    
    # Compute cutoff value cf
    cf = compute_cf(thr, tpr, roc_auc, auc_cf, tpr_cf)
    
    # Prepare data for plotting
    data = prepare_plot_data(final, neg_ind, hpos_ind)
    
    # Plot Probability Score Distribution
    plot_probability_score_distribution(data, minScore, out)
    
    # Plot ROC Curve
    plot_roc_curve(utd, tpr, thr, roc_auc, out)
    
    # Plot UTD Scatter
    plot_utd_scatter(final, neg_ind, minScore, cf, out)
    
    # Plot Germline Scatter
    plot_germline_scatter(final, hpos_ind, minScore, cf, out)
    
    return cf



def plot_probability_score_distribution(data, minScore, out):
    """Plot the probability score distribution for UTDs and Germline SNPs."""
    plt.rcParams['axes.linewidth'] = 3
    SNPs = ["UTDs", "Germline SNPs"]
    plt.figure(figsize=(10, 6))
    for SNP in SNPs:
        subset = data[data[:, 2] == SNP]
        sns.kdeplot(subset[:, 0].astype(float), shade=True, linewidth=3, clip=(0, 1), label=SNP)
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    plt.ylim([-0.1, ymax])
    plt.xlim([-0.02, 1.02])
    plt.xlabel('Probability score')
    plt.ylabel('Density')
    plt.plot([minScore, minScore], [0, ymax + 2], 'r--', linewidth=2)
    plt.legend(loc='upper right')
    plt.savefig(out + "_Probability_Score.png")
    plt.close()

def plot_roc_curve(utd, tpr, thr, roc_auc, out):
    """Plot the ROC curve and dropout variance threshold."""
    plt.figure(figsize=(10, 6))
    plt.plot(utd, tpr, label=f'ROC curve (area = {roc_auc:.2f})', linewidth=3)
    plt.xlim([0.0, np.amax(utd)])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Number of passed UTDs (False Positives)')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", prop={'size': 18})
    ax2 = plt.gca().twinx()
    ax2.plot(utd, thr, markeredgecolor='r', linestyle='dashed', color='r', linewidth=3)
    ax2.set_ylabel('Drop-out variance threshold', color='r')
    ax2.set_ylim([0.25, 0])
    ax2.set_xlim([0.0, np.amax(utd)])
    plt.savefig(out + "_ROC.png")
    plt.close()

def plot_utd_scatter(final, neg_ind, minScore, cf, out):
    """Plot scatter plot for UTDs showing uncertainty vs probability scores."""
    plt.figure(figsize=(10, 6))
    plt.scatter(final[neg_ind, 1], final[neg_ind, 0], c='grey', s=20, alpha=0.3)
    plt.xlim([-0.01, 0.21])
    plt.ylim([-0.01, 1.02])
    plt.plot([-0.01, 0.22], [minScore, minScore], 'r--', linewidth=2)
    plt.plot([cf, cf], [-0.2, 1.1], 'r--', linewidth=2)
    plt.xlabel('Uncertainty Score')
    plt.ylabel('Probability Score')
    plt.savefig(out + "_UTDs.png")
    plt.close()

def plot_germline_scatter(final, hpos_ind, minScore, cf, out):
    """Plot scatter plot for Germline SNPs showing uncertainty vs probability scores."""
    np.random.shuffle(final[hpos_ind])
    plt.figure(figsize=(10, 6))
    plt.scatter(final[hpos_ind[:1000], 1], final[hpos_ind[:1000], 0], c='b', s=20, alpha=0.3)
    plt.xlim([-0.01, 0.21])
    plt.ylim([-0.01, 1.02])
    plt.plot([-0.01, 0.22], [minScore, minScore], 'r--', linewidth=2)
    plt.plot([cf, cf], [-0.2, 1.1], 'r--', linewidth=2)
    plt.xlabel('Uncertainty Score')
    plt.ylabel('Probability Score')
    plt.savefig(out + "_Germline.png")
    plt.close()
