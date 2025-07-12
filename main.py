import typer

from models import ReportModel
from tokenizers import Tokenizer
from trainer import Trainer
from utils.arg_parser import parse_agrs

app = typer.Typer()




@app.command()
def train():
    split_frac = [0.8, 0.12, 0.08]
    tokenizer = Tokenizer(args.reports_json_path)
    model = ReportModel(args, tokenizer)
    trainer = Trainer(args, model, tokenizer, split_frac)
    trainer.train()
    print('model training finished')
    trainer.test()
    print('model testing finished')

    save_model(trainer)
    return trainer

@app.command()
def test(model_path):
    split_frac = [0.8, 0.12, 0.08]
    tokenizer = Tokenizer(args.reports_json_path)
    model = ReportModel.load_from_checkpoint(model_path, args=args, tokenizer=tokenizer)
    trainer = Trainer(args, model, tokenizer, split_frac)
    trainer.test()
    print('model testing finished')



def save_model(trainer):
    print(f'Saving model at path: {args.model_save_path}')
    trainer.save_model(args.model_save_path)





if __name__ == "__main__":
    args = parse_agrs()
    app()