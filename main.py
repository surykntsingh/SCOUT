import typer

from models import ReportModel
from report_tokenizers import Tokenizer
from trainer import Trainer, KFoldTrainer
from utils.arg_parser import parse_agrs
from utils.utils import save_model, get_params_for_key

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
def trainkfold(config_file_path='config.yaml', num_fold=3):
    args = get_params_for_key(config_file_path, "train")
    split_frac = [0.85, 0.15]
    tokenizer = Tokenizer(args.reports_json_path)
    # model = ReportModel(args, tokenizer)

    trainer = KFoldTrainer(args, tokenizer, split_frac, num_fold)
    trainer.train()





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




# args = parse_agrs()
if __name__ == "__main__":
    app()