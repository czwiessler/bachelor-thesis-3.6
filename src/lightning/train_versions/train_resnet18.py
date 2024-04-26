import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from src.lightning.model_architectures.model_coat_tiny import CoatTimmModel
from src.lightning.model_architectures.model_resnet18 import ResnetTimmModel
def init_trainer(model, use_lr_finder):
    # After defining this class, you can instantiate it and use it with a PyTorch Lightning Trainer as shown before.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='../../../models/checkpoints',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        filename='res18-{epoch}-{val_loss:.5f}'
    )


    trainer = pl.Trainer(max_epochs=100, log_every_n_steps=5, callbacks=[checkpoint_callback])
    if not use_lr_finder:
        return trainer
    if use_lr_finder:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        return trainer
    else:
        raise ValueError('use_lr_finder should be a boolean value')


model_instance = ResnetTimmModel(learning_rate=3.9810717055349735e-05)
trainer = init_trainer(model=model_instance, use_lr_finder=False)

trainer.fit(model_instance)
