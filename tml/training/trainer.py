import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from tml.models.model import BinaryClassificationLightning


def train_model(train_set, input_dim, batch_size, max_epochs, logger,
                 nb_classes=1, dropout_rate=None, learning_rate=1e-3):
    
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
                                          dropout_rate=dropout_rate,
                                          learning_rate=learning_rate
                                          )
    
    # checkpoint_monitor = 'train_loss'
    # checkpoint_mode = 'min'

    # checkpoint_callback = ModelCheckpoint(
    #     monitor=checkpoint_monitor,
    #     save_top_k=1,
    #     mode=checkpoint_mode,
    #     dirpath=f'{logger.save_dir}/{logger.name}/version_{logger.version}/checkpoints/',
    #     filename='vae-{epoch:02d}-{val_loss:.2f}'
    # )

    trainer = pl.Trainer(max_epochs=max_epochs,
                         enable_checkpointing=False,
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         logger=logger
                         )
    
    trainer.fit(model, train_loader)
    return model

