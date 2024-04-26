import pytorch_lightning as pl
from src.lightning.model_architectures.model_maxvit_base_384 import MaxVitTimmModel

def init_trainer(model, use_lr_finder):
    # After defining this class, you can instantiate it and use it with a PyTorch Lightning Trainer as shown before.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='../../../models/checkpoints',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        filename='maxVitBase384-{epoch}-{val_loss:.5f}'
    )


    trainer = pl.Trainer(max_epochs=10, log_every_n_steps=1, callbacks=[checkpoint_callback])
    return trainer


model_instance = MaxVitTimmModel()
trainer = init_trainer(model=model_instance, use_lr_finder=False)

trainer.fit(model_instance)
