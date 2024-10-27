import argparse

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import os

from tml.data.get_l1_l2 import (
    prepare_level1_training_set,
    prepare_level2_training_set,
    prune_test_set,
)
from tml.data.load_data import load_and_preprocess_data
from tml.models.model import BinaryClassificationLightning
from tml.plotting.plotting import tml_plots
from tml.training.trainer import train_model
from tml.utils.utils import load_config



# Main function to run the training and evaluation
def pipeline(config_path):
    
    config = load_config(config_path)

    # Initialize logger
    logger = TensorBoardLogger("logs", name=config['sample_name'])

    cols = range(1, config['num_cols']+1)
    config['input_dim'] = len(cols) - 1

    # Load data and normalize
    load_dict =load_and_preprocess_data(config['input_path'], cols)
    out1 = load_dict['out1']
    out2_var = load_dict['out2_var']

    for cnt in range(config['sampling_num']):
        np.random.seed(cnt + 1)
        # Prepare Level 1 training data
        train_set_L1 = prepare_level1_training_set(load_dict['pos_ind'],
                                                   load_dict['neg_ind'],
                                                   load_dict['all_set'],
                                                   cnt
                                                   )

        # Level 1 training
        print(f"Level 1 training: subset {cnt+1}")
        model_L1, _ = train_model(
            train_set=train_set_L1,
            input_dim=config['input_dim'],
            nb_classes=config['nb_classes']-1,
            batch_size=config['batch_size'],
            max_epochs=config['epochs'],
            # dropout_rate=None,
            learning_rate=config['learning_rate'],
            logger=logger
        )

        # Level 1 testing
        print(f"Level 1 test: subset {cnt+1}")
        y_pred = model_L1.predict(load_dict['test_set'][:, 1:])

        # Pruning
        TPs, TNs = prune_test_set(load_dict['test_set'],
                                  y_pred,
                                  lower_thresh=config['lower_threshold'],
                                  upper_thresh=config['upper_threshold'])

        # Prepare Level 2 training data
        train_set_L2 = prepare_level2_training_set(TPs, TNs, load_dict['test_set'], cnt)

        # Level 2 training
        print(f"Level 2 training: subset {cnt+1}")
        model_L2, checkpoint_path_L2 = train_model(
            train_set=train_set_L2,
            input_dim=config['input_dim'],
            nb_classes=config['nb_classes']-1,
            batch_size=config['batch_size'],
            max_epochs=config['epochs'],
            # dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate'],
            logger=logger,
            cnt=cnt+1,
            # save=True
        )

        # Ensure out1 has two dimensions
        # print(f"out1 dims: {out1.ndim}")
        if out1.ndim == 1:
            out1 = out1.reshape(-1, 1)

        # Level 2 testing
        print(f"Level 2 test: subset {cnt+1}")
        y_pred_all = model_L2.predict(load_dict['all_set'][:, 1:])
        y_pred_all = y_pred_all.reshape(-1, 1)
        out1 = np.hstack((out1, y_pred_all))

        # Predict with dropout

        # model_T2 = BinaryClassificationLightning.load_from_checkpoint(
        #     checkpoint_path_L2,
        #     input_dim=config['input_dim'],
        #     dropout_rate=config['dropout_rate'],
        #     learning_rate=config['learning_rate'],
        # )

        # y_pred_dropout = model_T2.predict_with_dropout(load_dict['all_set'][:, 1:], config['drop_it'])

        print(f"Level 2 test with dropout: subset {cnt+1}")
        y_pred_dropout = model_L2.predict_with_dropout(load_dict['all_set'][:, 1:], config['drop_it'])

        if out2_var.ndim == 1:
            out2_var = out2_var.reshape(-1, 1)
        
        y_pred_var = np.var(y_pred_dropout, axis=0).reshape(-1, 1)
        out2_var = np.hstack((out2_var, y_pred_var))

    # You can now use out1 and out2_var as needed

    # Calculate final results and save
    print("Calculate final results")
    y_pred_mean_1 = np.mean(out1[:, 1:], axis=1)
    y_pred_mvar = np.mean(out2_var[:, 1:], axis=1)
    final = np.column_stack((y_pred_mean_1, y_pred_mvar))
    

    # Plot and calculate thresholds
    os.makedirs(config['out_path'], exist_ok=True)
    

    thr = tml_plots(final,
                    load_dict['neg_ind'],
                    load_dict['hpos_ind'],
                    config['pscore_cf'],
                    config['auc_cf'],
                    config['tpr_cf'],
                    f"{config['out_path']}/{config['sample_name']}"
                    )

    final = np.column_stack((final, np.repeat("PASS", len(y_pred_mvar))))
    final[final[:,1].astype(float) > thr, 2] = "FAIL_Uncertain"
    final[final[:,0].astype(float) <= config['pscore_cf'], 2] = "FAIL_LowScore"
    save = np.column_stack((load_dict['names'], final))
    header = ['Mutation', 'Type', 'Probability_Score', 'Uncertainty_Score', 'Result']
    pd.DataFrame(save.astype(str)).to_csv(f"{config['out_path']}/{config['sample_name']}_scores.csv", header=header, index=None)



if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="tml/configs/config.yaml", help='Path to config file.')
    args = parser.parse_args()

    pipeline(args.config)
    