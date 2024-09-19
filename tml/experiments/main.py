import argparse
import yaml
import pytorch_lightning as pl
from training.trainer import tmlTrainer
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/config.yaml", help='Path to config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize logger
    logger = TensorBoardLogger("logs", name=config['sample_name'])

    # Initialize model and trainer
    model = MutLXTrainer(input_dim=42, nb_classes=2)
    trainer = pl.Trainer(max_epochs=config['epochs'], logger=logger)

    # Train model
    trainer.fit(model)
