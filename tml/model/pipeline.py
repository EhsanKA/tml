
from tml.data.load_data import load_and_preprocess_data
from tml.data.get_l1_l2 import (
    prepare_level1_training_set,
    prepare_level2_training_set,
    prune_test_set,
)
from tml.training.trainer import train_model


from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np


class Pipeline():
    def __init__(self,
                 model,
                 data,
                 hard_targets,
                 soft_targets,
                 logger=None,
                 seed=42,

                 ):
        
        self.model = model
        self.data = data
        self.hard_targets = hard_targets
        # self.soft_targets = soft_targets
        # self.probability_scores = None
        # self.uncertainty_scores = None

        if logger is None:
            self.logger = TensorBoardLogger("logs", name="default")


        self.cols = range(1, 42+1)
        self.input_dim = len(self.cols) - 1
        self.load_dict = load_and_preprocess_data(config['input_path'], self.cols)

        self.nb_classes=2-1
        self.batch_size=64
        self.max_epochs=100
        self.learning_rate=1e-3
    
        self.out1 = self.load_dict['out1']
        self.out2_var = self.load_dict['out2_var']
        # self.dataset = self.load_dict['test_set']
        # self.data = self.dataset[:, 1:]
        # self.labels = self.dataset[:, 0]
        self.seed = seed

        self.lower_threshold = 0.3
        self.upper_threshold = 0.7

        self.drop_iterations = 25


    def run(self, n_steps=1):

        for i in range(n_steps):
            self.inc_seed()
            self.init_level1()
            self.train_level1()
            self.predict_level1()
            self.prunning()
            self.init_level2()
            self.train_level2()
            self.predict_level2()
            self.predict_level2_with_dropout()
            self.get_probability_scores()
            self.get_uncertainty_scores()
            


    def inc_seed(self):
        self.seed += 1

    
    def init_level1(self):
        self.train_set_L1 = prepare_level1_training_set(self.load_dict['pos_ind'],
                                                        self.load_dict['neg_ind'],
                                                        self.load_dict['all_set'],
                                                        self.seed
                                                        )

    def train_level1(self):
        self.model_L1, _ = train_model(
            train_set=self.train_set_L1,
            input_dim=self.input_dim,
            nb_classes=self.nb_classes,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            learning_rate=self.learning_rate,
            logger=self.logger
        )

    def predict_level1(self):
        print(f"Level 1 test: subset {self.seed}")
        self.y_pred = self.model_L1.predict(self.data)


    def prunning(self):
        self.TPs, self.TNs = prune_test_set(self.data,
                                            self.y_pred,
                                            lower_thresh=self.lower_threshold,
                                            upper_thresh=self.upper_threshold)

    def init_level2(self):
        self.train_set_L2 = prepare_level2_training_set(self.TPs,
                                                        self.TNs,
                                                        self.data,
                                                        self.seed)


    def train_level2(self):
        self.model_L2, self.checkpoint_path_L2 = train_model(
            train_set=self.train_set_L2,
            input_dim=self.input_dim,
            nb_classes=self.nb_classes,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            learning_rate=self.learning_rate,
            logger=self.logger,
            cnt=self.seed,
        )

    def predict_level2(self):
        if self.out1.ndim == 1:
            self.out1 = self.out1.reshape(-1, 1)

        print(f"Level 2 test: subset {self.seed}")
        self.y_pred_all = self.model_L2.predict(self.data)
        self.y_pred_all = self.y_pred_all.reshape(-1, 1)
        self.out1 = np.hstack((self.out1, self.y_pred_all))

    def predict_level2_with_dropout(self):
        print(f"Level 2 test with dropout: subset {self.seed}")
        self.y_pred_dropout = self.model_L2.predict_with_dropout(self.data,
                                                                 self.drop_iterations
                                                                 )

        if self.out2_var.ndim == 1:
            self.out2_var = self.out2_var.reshape(-1, 1)
        
        self.y_pred_var = np.var(self.y_pred_dropout, axis=0).reshape(-1, 1)
        self.out2_var = np.hstack((self.out2_var, self.y_pred_var))

    def get_probability_scores(self):
        self.y_pred_mean_1 = np.mean(self.out1[:, 1:], axis=1)
        self.probability_scores = self.y_pred_mean_1


    def get_uncertainty_scores(self):
        self.y_pred_var_mean = np.mean(self.out2_var[:, 1:], axis=1)
        self.uncertainty_scores = self.y_pred_var_mean
