import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tml.data.prep_typebased import prep_typebased

def load_and_preprocess_data(input_path, cols):
    # Load data
    prep_dict = prep_typebased(input_path, cols)
    
    all_set = prep_dict["all_set"]
    neg_ind = prep_dict["neg_ind"]
    pos_ind = prep_dict["pos_ind"]

    # Apply standardization
    scaler = StandardScaler()
    test_set = all_set[np.sort(np.concatenate((pos_ind, neg_ind)))]
    ## todo: why do we fit_transform on test_set?
    test_set[:, 1:] = scaler.fit_transform(test_set[:, 1:])
    all_set[:, 1:] = scaler.transform(all_set[:, 1:])

    prep_dict['all_set'] = all_set
    prep_dict['out1'] = test_set[:, 0]
    prep_dict['out2_var'] = test_set[:, 0]
    prep_dict['test_set'] = test_set


    return prep_dict
