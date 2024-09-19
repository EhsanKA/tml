import pandas as pd
import numpy as np

def load_and_preprocess_data(input_path, cols):
    # Load data from CSV
    df = pd.read_csv(input_path)
    all_set, test_ind, neg_ind, pos_ind, hpos_ind, names = prep_typebased(df, cols)
    
    # Apply standardization
    scaler = StandardScaler()
    all_set[:, 1:] = scaler.fit_transform(all_set[:, 1:])
    return all_set, test_ind, neg_ind, pos_ind, hpos_ind, names
