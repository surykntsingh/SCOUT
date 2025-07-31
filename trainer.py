import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from datamodules.wsi_embedding_datamodule import PatchEmbeddingDataModule, PatchEmbeddingDataPredictModule


class Trainer:

    def __init__(self, args, model, tokenizer, split_frac=(0.75, 0.12, 0.13), predict=False, slide_ids=None):
        self.ckpt_path = args.ckpt_path
        self.max_epochs = args.max_epochs
        self.split_frac = split_frac
        if predict:
            self.datamodule = PatchEmbeddingDataPredictModule(args, tokenizer, slide_ids=slide_ids)
        else:
            self.datamodule = PatchEmbeddingDataModule(args, tokenizer, split_frac)
        self.model = model
        pl.seed_everything(42)
        self.trainer = None
        self.__devices = list(map(int, args.devices.split(',')))

    def train(self, fast_dev_run=False):
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.ckpt_path,  # Directory to save checkpoints
            filename="best_model",  # Naming convention
            monitor="val_loss",  # Metric to monitor for saving best checkpoints
            mode="min",  # Whether to minimize or maximize the monitored metric
            save_top_k=1,  # Number of best checkpoints to keep
            save_last=True  # Save the last checkpoint regardless of the monitored metric
        )
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            accelerator='gpu',
            devices=self.__devices,
            strategy='ddp_find_unused_parameters_true',
            enable_progress_bar=True,
            log_every_n_steps=2,
            fast_dev_run=fast_dev_run
        )
        self.trainer.fit(
            self.model, datamodule=self.datamodule
        )
        train_metrics = self.trainer.logged_metrics
        return train_metrics

    def test(self, fast_dev_run=False):

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=self.__devices,
            strategy='ddp_find_unused_parameters_true',
            enable_progress_bar=True,
            log_every_n_steps=2,
            fast_dev_run=fast_dev_run
        )

        trainer.test(
            self.model, datamodule=self.datamodule
        )
        test_metrics = trainer.logged_metrics
        return test_metrics

    def predict(self, fast_dev_run=False):


        trainer = pl.Trainer(
            accelerator='gpu',
            devices=self.__devices,
            strategy='ddp_find_unused_parameters_true',
            enable_progress_bar=True,
            log_every_n_steps=2,
            fast_dev_run=fast_dev_run
        )

        preds = trainer.predict(
            self.model, datamodule=self.datamodule
        )
        return preds




    @rank_zero_only
    def save_model(self, model_path):

        self.trainer.save_checkpoint(model_path)
        print(f'model saved at path: {model_path}')

    def load_model(self, model_cls, model_path, **kwargs):
        return model_cls.load_from_checkpoint(model_path, **kwargs)
