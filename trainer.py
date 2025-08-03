import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from datamodules.wsi_embedding_datamodule import PatchEmbeddingDataModule, EmbeddingDataModule

from sklearn.model_selection import KFold

from models import ReportModel
from utils.utils import read_json_file


class Trainer:

    def __init__(self, args, tokenizer, split_frac):
        self.ckpt_path = args.ckpt_path
        self.max_epochs = args.max_epochs
        self.split_frac = split_frac
        # self.datamodule = PatchEmbeddingDataModule(args, tokenizer, split_frac)
        # self.model = ReportModel(args, tokenizer)
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

        trainer.predict(
            self.model, datamodule=self.datamodule
        )
        test_metrics = trainer.logged_metrics
        return test_metrics


    @rank_zero_only
    def save_model(self, model_path):

        self.trainer.save_checkpoint(model_path)
        print(f'model saved at path: {model_path}')

    def load_model(self, model_cls, model_path, **kwargs):
        return model_cls.load_from_checkpoint(model_path, **kwargs)


class KFoldTrainer(Trainer):
    def __init__(self, args, tokenizer, split_frac):
        super().__init__(args, tokenizer, split_frac)
        self.__reports = read_json_file(args.reports_json_path)
        # self.__slides = reports.keys()
        self.__kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)
        self.args = args
        self.tokenizer = tokenizer
        self.split_frac =split_frac

    def train(self, fast_dev_run=False):

        for fold, (train_idx, test_idx) in enumerate(self.__kf.split(self.__reports)):
            self.datamodule = EmbeddingDataModule(self.args, self.tokenizer, self.split_frac, train_idx, test_idx)
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.ckpt_path,  # Directory to save checkpoints
                filename=f"fold{fold}_" + "{epoch:02d}_{val_loss:.5f}",  # Naming convention
                monitor="val_loss",  # Metric to monitor for saving best checkpoints
                mode="min",  # Whether to minimize or maximize the monitored metric
                save_top_k=1,  # Number of best checkpoints to keep
                save_last=True  # Save the last checkpoint regardless of the monitored metric
            )
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True,
                                                mode="min")
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                callbacks=[checkpoint_callback, early_stop_callback],
                accelerator='gpu',
                devices=self.__devices,
                strategy='ddp_find_unused_parameters_true',
                enable_progress_bar=True,
                log_every_n_steps=2,
                fast_dev_run=fast_dev_run
            )

            model = ReportModel(self.args, self.tokenizer, reports=self.__reports)
            trainer.fit(
                model, datamodule=self.datamodule
            )
            train_metrics = self.trainer.logged_metrics

            print(f'fold: {fold}, train_metrics: {train_metrics}')

            trainer.test(
                self.model, datamodule=self.datamodule
            )
            test_metrics = trainer.logged_metrics

            print(f'fold: {fold}, test_metrics: {test_metrics}')



