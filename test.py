from models import ReportModel
from report_tokenizers import Tokenizer
from trainer import Trainer
from utils.arg_parser import parse_agrs
from utils.utils import save_model


def test(args):
    split_frac = [0.8, 0.12, 0.08]
    tokenizer = Tokenizer(args.reports_json_path)
    model_path = args.model_save_path

    model = ReportModel.load_from_checkpoint(model_path, args=args, tokenizer=tokenizer)
    trainer = Trainer(args, model, tokenizer, split_frac)

    test_metrics = trainer.test()
    print('model testing finished')

    print(f'test_metrics: {test_metrics}')


if __name__ == "__main__":
    args = parse_agrs()
    test(args)
    print('Finished processing!')