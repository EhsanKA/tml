import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from tml.models.model import BinaryClassificationLightning


def train_model(train_set, input_dim, batch_size, max_epochs, logger,
                nb_classes=1, dropout_rate_train=0.0, dropout_rate_pred=0.8,   
                learning_rate=1e-3, cnt=0, save=False
                ):
    
    X_train = train_set[:, 1:]
    y_train = train_set[:, 0]
    dataset = TensorDataset(torch.from_numpy(X_train).float(), 
                            torch.from_numpy(y_train).float())
    
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True
                              )

    model = BinaryClassificationLightning(input_dim=input_dim,
                                          nb_classes=nb_classes,
                                          dropout_rate_train=dropout_rate_train,
                                          dropout_rate_pred=dropout_rate_pred,
                                          learning_rate=learning_rate
                                          )

    trainer = pl.Trainer(max_epochs=max_epochs,
                         enable_checkpointing=False,
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         logger=logger
                         )
    
    trainer.fit(model, train_loader)

    # Save the final model checkpoint
    checkpoint_path = f'{logger.save_dir}/{logger.name}/version_{logger.version}/checkpoints/model_{cnt}.ckpt'
    if save:
        trainer.save_checkpoint(checkpoint_path)

    return model, checkpoint_path

