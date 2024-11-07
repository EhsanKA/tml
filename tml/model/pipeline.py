
from pytorch_lightning.loggers import TensorBoardLogger


class Pipeline():
    def __init__(self,
                 model,
                 dataset,
                 hard_targets,
                 soft_targets,
                 logger=None,
                 seed=42,

                 ):
        
        self.model = model
        self.dataset = dataset
        self.hard_targets = hard_targets
        self.soft_targets = soft_targets
        self.probability_scores = None
        self.uncertainty_scores = None

        if logger is None:
            self.logger = TensorBoardLogger("logs", name="default")



    def inc_seed(self):
        self.seed += 1

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
            

    def init_level1(self):
        pass

    def train_level1(self):
        pass

    def predict_level1(self):
        pass

    def prunning(self):
        pass

    def init_level2(self):
        pass

    def train_level2(self):
        pass

    def predict_level2(self):
        pass

    def predict_level2_with_dropout(self):
        pass

    def get_probability_scores(self):
        pass

    def get_uncertainty_scores(self):
        pass
