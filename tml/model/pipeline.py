
# from tml.data.load_data import load_and_preprocess_data
# from tml.data.get_l1_l2 import (
#     prepare_level1_training_set,
#     prepare_level2_training_set,
#     prune_test_set,
# )
# from tml.training.trainer import train_model
from tml.model.utils import get_input_shape, ensure_tensors
from tml.model.tml_dataset import BalancedSampler, TMLDataset, prune

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

class ModelHandler:
    def __init__(self, model_instance=None, model_class=None, model_config=None):
        """
        model_instance : Initialized model instance provided by the user.
        model_class    : Model class reference, used to initialize new instances.
        model_config   : Dictionary of hyperparameters (if needed for re-initialization).
        """

        if model_instance is None and model_class is None:
            raise ValueError("At least one of model_instance or model_class must be provided.")
        
        if model_instance:
            self.model_class = type(model_instance)  # Use class of model_instance if not provided
            self.model_config = getattr(model_instance, 'hparams', {})
        else:
            if not self.model_config and model_class:
                raise ValueError("model_config and model_class must be provided if model_instance is not.")
            self.model_class = model_class
            self.model_config = model_config



    def initialize_new_model(self):
        """
        Initialize a new model instance using the model class and configuration.
        """
        # Instantiate a new model using the model class and provided hyperparameters
        new_model = self.model_class(**self.model_config)
        return new_model



class Pipeline():
    def __init__(self,
                 model_handler,
                 data,
                 hard_targets,
                 batch_size=64,
                 max_epochs=1,
                 learning_rate=1e-3,
                 lower_threshold=0.3,
                 upper_threshold=0.7,
                 drop_iterations=2,
                 logger=None,
                 logger_name="default",
                 seed=42,
                #  out1=None,
                #  out2_var=None,
                 ):
        
        self.model_handler = model_handler
        # data and hard_targets are tensors
        self.data, self.hard_targets = ensure_tensors(data, hard_targets)

        self.probability_scores = None
        self.uncertainty_scores = None


        self.input_dim, self.n_samples = get_input_shape(self.data)
        # self.load_dict = load_and_preprocess_data(config['input_path'], self.cols)

        # self.nb_classes=2-1
        self.batch_size=batch_size
        self.max_epochs= max_epochs
        self.learning_rate= learning_rate
    
        # self.out1 = out1
        # self.out2_var = out2_var


        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.drop_iterations = drop_iterations
        self.dataset = TMLDataset(self.data, self.hard_targets)
        self.seed = seed

        if logger is None:
            self.logger = TensorBoardLogger("logs", name=logger_name)


    def run(self, n_steps=1):
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False
                                      )
        
        self.all_probs = np.zeros((n_steps, self.n_samples))
        self.all_vars = np.zeros((n_steps, self.n_samples))

        for i in range(n_steps):
            self.step = i
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
        print(f"Seed: {self.seed}")
        self.seed += 1

    def init_level1(self):
        print(f"Level 1 init: subset {self.seed}")
        self.balanced_sampler = BalancedSampler(self.dataset, seed=self.seed)
        self.balanced_dataset = self.balanced_sampler.sample_balanced_subset()

        self.train_set_L1 = self.balanced_dataset
        # self.train_set_L1 = prepare_level1_training_set(self.load_dict['pos_ind'],
        #                                                 self.load_dict['neg_ind'],
        #                                                 self.load_dict['all_set'],
        #                                                 self.seed
        #                                                 )

    def train_level1(self):
        print(f"Level 1 training: subset {self.seed}")
        self.model = self.model_handler.initialize_new_model()
        train_loader = DataLoader(self.train_set_L1,
                                  batch_size=self.batch_size,
                                  shuffle=True
                                  )
        
        trainer = pl.Trainer(max_epochs=self.max_epochs,
                             enable_checkpointing=False,
                             accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                             logger=self.logger
                             )
    
        trainer.fit(self.model, train_loader)
    
        # self.model_L1, _ = train_model(
        #     train_set=self.train_set_L1,
        #     input_dim=self.input_dim,
        #     nb_classes=self.nb_classes,
        #     batch_size=self.batch_size,
        #     max_epochs=self.max_epochs,
        #     learning_rate=self.learning_rate,
        #     logger=self.logger
        # )

    def predict_level1(self):
        print(f"Level 1 predict: subset {self.seed}")
        self.y_pred = self.model.predict(self.data_loader)

    def prunning(self):
        print(f"Pruning: subset {self.seed}")
        self.TPs, self.TNs = prune(self.hard_targets,
                                   self.y_pred,
                                   lower_thresh=self.lower_threshold,
                                   upper_thresh=self.upper_threshold)

    def init_level2(self):
        print(f"Level 2 init: subset {self.seed}")
        self.balanced_sampler = BalancedSampler(self.dataset, indices= self.TPs + self.TNs,
                                                seed=self.seed)

        self.train_set_L2 = self.balanced_sampler.sample_balanced_subset()
        # self.train_set_L2 = prepare_level2_training_set(self.TPs,
        #                                                 self.TNs,
        #                                                 self.data,
        #                                                 self.seed)

    def train_level2(self):
        print(f"Level 2 training: subset {self.seed}")
        self.model = self.model_handler.initialize_new_model()

        train_loader = DataLoader(self.train_set_L2,
                                  batch_size=self.batch_size,
                                  shuffle=True
                                  )
        
        trainer = pl.Trainer(max_epochs=self.max_epochs,
                             enable_checkpointing=False,
                             accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                             logger=self.logger
                             )
    
        trainer.fit(self.model, train_loader)

        # self.model_L2, self.checkpoint_path_L2 = train_model(
        #     train_set=self.train_set_L2,
        #     input_dim=self.input_dim,
        #     nb_classes=1,
        #     batch_size=self.batch_size,
        #     max_epochs=self.max_epochs,
        #     learning_rate=self.learning_rate,
        #     logger=self.logger,
        #     cnt=self.seed,
        # )

    def predict_level2(self):
        print(f"Level 2 predict: subset {self.seed}")
        # if self.out1.ndim == 1:
        #     self.out1 = self.out1.reshape(-1, 1)

        self.y_pred_all = self.model.predict(self.data_loader)
        self.all_probs[self.step] = self.y_pred_all
        # self.y_pred_all = self.y_pred_all.reshape(-1, 1)
        # self.out1 = np.hstack((self.out1, self.y_pred_all))

    def predict_level2_with_dropout(self):
        print(f"Level 2 prediction with dropout: subset {self.seed}")
        self.y_pred_dropout = self.predict_with_dropout(self.data_loader,
                                                        self.drop_iterations
                                                        )

        # if self.out2_var.ndim == 1:
        #     self.out2_var = self.out2_var.reshape(-1, 1)
        # self.y_pred_var = np.var(self.y_pred_dropout, axis=0).reshape(-1, 1)
        # self.out2_var = np.hstack((self.out2_var, self.y_pred_var))
        self.all_vars[self.step] = np.var(self.y_pred_dropout, axis=0)
        

    def get_probability_scores(self):
        print(f"Getting probability scores: subset {self.seed}")
        # self.y_pred_mean_1 = np.mean(self.out1[:, 1:], axis=1)
        # self.probability_scores = self.y_pred_mean_1
        self.probability_scores = np.mean(self.all_probs, axis=0)


    def get_uncertainty_scores(self):
        print(f"Getting uncertainty scores: subset {self.seed}")
        self.y_pred_var_mean = np.mean(self.all_vars, axis=0)
        self.uncertainty_scores = self.y_pred_var_mean


    def predict_with_dropout(self, dataloader, n_iter=10):
        """
        Predict with dropout active for uncertainty estimation (MC Dropout).
        
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader containing input data.
            n_iter (int, optional): Number of stochastic forward passes. Defaults to 10.
            
        Returns:
            numpy.ndarray: Predicted probabilities from multiple forward passes, with shape [n_iter, total_samples].
        """
        self.model.eval()  # Ensure layers like BatchNorm are in eval mode
        self.enable_dropout()  # Activate dropout layers
        
        # Pre-allocate a tensor for all predictions
        device = next(self.model.parameters()).device  # Get model device (CPU or GPU)
        all_preds = torch.zeros((n_iter, self.n_samples), device=device)  # Use model's device

        # Perform multiple stochastic forward passes
        for i in range(n_iter):
            batch_preds = []
            with torch.no_grad():
                for batch in dataloader:
                    X_batch = batch[0] if isinstance(batch, (list, tuple)) else batch  # Handle case where batch is (X, y)
                    X_batch = X_batch.to(device).float()  # Move input to model's device and ensure float32
                    y_pred = self.model(X_batch)
                    batch_preds.append(y_pred)  # Keep as tensor on the model's device

            # Concatenate predictions and store in all_preds for the current iteration
            all_preds[i] = torch.cat(batch_preds, dim=0).squeeze()  # Shape: [n_samples]

        self.disable_dropout()  # Disable dropout after prediction

        # Convert all_preds to numpy at the end, after all iterations are done
        return all_preds.cpu().numpy()  # Shape: [n_iter, total_samples]

    

    def enable_dropout(self):
        """Enable dropout layers for MC Dropout without affecting other layers."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()  # Enable dropout only

    def disable_dropout(self):
        """Disable dropout layers after MC Dropout."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()  # Set dropout back to eval mode
