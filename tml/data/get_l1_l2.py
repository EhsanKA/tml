import numpy as np

# Function to prepare Level 1 training data
def prepare_level1_training_set(pos_ind, neg_ind, all_set, cnt):
    np.random.seed(cnt + 1)
    pos_ind_subset = np.random.choice(pos_ind, size=len(neg_ind), replace=False)
    indices = np.sort(np.concatenate((pos_ind_subset, neg_ind)))
    train_set = all_set[indices]
    return train_set

# Function to prune the test set
def prune_test_set(test_set, y_pred, lower_thresh=0.3, upper_thresh=0.7):
    labels = test_set[:, 0]
    TPs = [ind for ind, (label, pred) in enumerate(zip(labels, y_pred)) if label == 1 and pred > lower_thresh]
    TNs = [ind for ind, (label, pred) in enumerate(zip(labels, y_pred)) if label == 0 and pred < upper_thresh]
    return TPs, TNs

# Function to prepare Level 2 training data
def prepare_level2_training_set(TPs, TNs, test_set, cnt):
    np.random.seed(cnt + 1)
    TPs_subset = np.random.choice(TPs, size=len(TNs), replace=False)
    indices = np.sort(np.concatenate((TPs_subset, TNs)))
    train_set = test_set[indices]
    return train_set