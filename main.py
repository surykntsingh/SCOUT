import typer

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
    split_frac = [0.8, 0.12, 0.08]
    tokenizer = Tokenizer(args.reports_json_path)
    model = ReportModel(args, tokenizer)
    trainer = Trainer(args, model, tokenizer, split_frac)
    train_metrics = trainer.train()
    print('model training finished')
    test_metrics = trainer.test()
    print('model testing finished')
    save_model(args, trainer)

    print(f'train_metrics: {train_metrics}, test_metrics: {test_metrics}')
    return trainer

@app.command()
def trainkfold(config_file_path='config.yaml'):
    args = get_params_for_key(config_file_path, "train")
    split_frac = [0.85, 0.15]
    tokenizer = Tokenizer(args.reports_json_path)
    # model = ReportModel(args, tokenizer)

    trainer = KFoldTrainer(args, tokenizer, split_frac)
    trainer.train(fast_dev_run=args.fast_dev_run)

    train_metrics, test_metrics = trainer.get_metrics()
    print(f'train_metrics: {train_metrics}')
    print(f'test_metrics: {test_metrics}')
    write_metrics(f'{args.results_path}/experiments/results.csv', {**train_metrics, **test_metrics})




@app.command()
def test(config_file_path='config.yaml'):
    model_path = 'saved_models/mi_gen_titan_plus_conch1.5_20250711_1_model.ckpt'
    args = get_params_for_key(config_file_path, "test")
    split_frac = [0.8, 0.12, 0.08]
    tokenizer = Tokenizer(args.reports_json_path)
    model = ReportModel.load_from_checkpoint(model_path, args=args, tokenizer=tokenizer)
    trainer = Trainer(args, model, tokenizer, split_frac)
    trainer.test()
    print('model testing finished')


def write_metrics(results_path, metrics):
    metrics['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_df = pd.DataFrame([metrics])


    metrics_df.to_csv(results_path,mode='a')





# args = parse_agrs()
if __name__ == "__main__":
    app()