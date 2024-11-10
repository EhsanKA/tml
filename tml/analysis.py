import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, brier_score_loss
from sklearn.calibration import calibration_curve

class BinaryClassificationAnalysis:
    def __init__(self, labels, probability_scores, uncertainty_scores):
        self.labels = np.array(labels)
        self.probability_scores = np.array(probability_scores)
        self.uncertainty_scores = np.array(uncertainty_scores)

    ### Metric Calculation Functions ###

    def calculate_roc_auc(self):
        fpr, tpr, _ = roc_curve(self.labels, self.probability_scores)
        return auc(fpr, tpr)

    def calculate_precision_recall(self):
        precision, recall, _ = precision_recall_curve(self.labels, self.probability_scores)
        return precision, recall

    def calculate_brier_score(self):
        return brier_score_loss(self.labels, self.probability_scores)

    def expected_calibration_error(self, n_bins=10):
        prob_true, prob_pred = calibration_curve(self.labels, self.probability_scores, n_bins=n_bins)
        return np.mean(np.abs(prob_true - prob_pred))

    ### Thresholding Methods ###

    def find_optimal_threshold(self):
        fpr, tpr, thresholds = roc_curve(self.labels, self.probability_scores)
        optimal_idx = np.argmax(tpr - fpr)  # Youden's Index
        return thresholds[optimal_idx]

    def uncertainty_based_filter(self, threshold):
        return self.probability_scores[self.uncertainty_scores > threshold]

    ### Plotting Methods ###

    def plot_reliability_diagram(self, n_bins=10):
        prob_true, prob_pred = calibration_curve(self.labels, self.probability_scores, n_bins=n_bins)
        plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.show()

    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.labels, self.probability_scores)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.calculate_roc_auc():.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    def plot_uncertainty_distribution(self):
        plt.hist(self.uncertainty_scores, bins=20, alpha=0.7, color='blue')
        plt.xlabel('Uncertainty Score')
        plt.ylabel('Frequency')
        plt.title('Uncertainty Score Distribution')
        plt.show()

    def plot_uncertainty_vs_confidence(self):
        plt.scatter(self.uncertainty_scores, self.probability_scores, alpha=0.5, color='purple')
        plt.xlabel('Uncertainty Score')
        plt.ylabel('Predicted Probability')
        plt.title('Uncertainty vs Confidence')
        plt.show()
