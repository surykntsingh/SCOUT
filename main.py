import typer

from datamodules.wsi_embedding_datamodule import PatchEmbeddingDataModule
from models import ReportModel
from report_tokenizers import Tokenizer
from trainer import Trainer, KFoldTrainer
import pandas as pd
from utils.utils import save_model, get_params_for_key
from datetime import datetime

app = typer.Typer()




@app.command()
def train(config_file_path='config.yaml'):
    args = get_params_for_key(config_file_path, "train")
    split_frac = [0.8, 0.14, 0.06]
    tokenizer = Tokenizer(args.reports_json_path)
    model = ReportModel(args, tokenizer)

    datamodule = PatchEmbeddingDataModule(args, tokenizer, split_frac)
    trainer = Trainer(args, tokenizer, split_frac)
    train_metrics = trainer.train(model, datamodule)
    print('model training finished')
    print(f'loading best model from {trainer.best_model_path}' )
    model = ReportModel.load_from_checkpoint(trainer.best_model_path, args=args, tokenizer=tokenizer)
    test_metrics = trainer.test(model, datamodule)
    print('model testing finished')
    # save_model(args, trainer)
    metrics = {**train_metrics, **test_metrics, 'best_model_path': trainer.best_model_path}
    print(f'train_metrics: {train_metrics}, test_metrics: {test_metrics}')

    write_metrics(f'{args.results_path}/experiments/results.csv', metrics)

@app.command()
def trainkfold(config_file_path='config.yaml'):
    args = get_params_for_key(config_file_path, "train")
    split_frac = [0.85, 0.15]
    tokenizer = Tokenizer(args.reports_json_path)
    # model = ReportModel(args, tokenizer)

    trainer = KFoldTrainer(args, tokenizer, split_frac)
    trainer.train_and_test(fast_dev_run=args.fast_dev_run)

    metrics = trainer.get_metrics()
    print(f'metrics: {metrics}')
    write_metrics(f'{args.results_path}/experiments/results.csv', metrics)




@app.command()
def test(config_file_path='config.yaml'):

    args = get_params_for_key(config_file_path, "train")
    split_frac = [0.8, 0.12, 0.08]
    tokenizer = Tokenizer(args.reports_json_path)
    datamodule = PatchEmbeddingDataModule(args, tokenizer, split_frac)
    trainer = Trainer(args, tokenizer, split_frac)

    print(f'loading best model from {args.model_load_path}')
    model = ReportModel.load_from_checkpoint(trainer.best_model_path, args=args, tokenizer=tokenizer)
    test_metrics = trainer.test(model, datamodule)
    print(f'test_metrics: {test_metrics}')
    print('model testing finished')


def write_metrics(results_path, metrics):
    metrics['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_df = pd.DataFrame([metrics])


    metrics_df.to_csv(results_path,mode='a')





# args = parse_agrs()
if __name__ == "__main__":
    app()