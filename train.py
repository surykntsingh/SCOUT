from models import ReportModel
from report_tokenizers import Tokenizer
from trainer import Trainer
from utils.arg_parser import parse_agrs
from utils.utils import save_model, read_json_file


def train(args):
    split_frac = [0.8, 0.12, 0.08]
    tokenizer = Tokenizer(args.reports_json_path)
    model = ReportModel(args, tokenizer, reports = read_json_file(args.reports_json_path))
    trainer = Trainer(args, model, tokenizer, split_frac)
    train_metrics = trainer.train()
    print('model training finished')
    test_metrics = trainer.test()
    print('model testing finished')
    save_model(args, trainer)

    print(f'train_metrics: {train_metrics}, test_metrics: {test_metrics}')


if __name__ == "__main__":
    args = parse_agrs()
    train(args)
    print('Finished processing!')