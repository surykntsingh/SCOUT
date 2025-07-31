from models import ReportModel
from report_tokenizers import Tokenizer
from trainer import Trainer
from utils.arg_parser import parse_agrs
from utils.utils import save_model



def test(args):

    tokenizer = Tokenizer(args.reports_json_path)
    model_path = args.model_save_path

    model = ReportModel.load_from_checkpoint(model_path, args=args, tokenizer=tokenizer)
    slide_ids = []
    trainer = Trainer(args, model, tokenizer, predict=True, slide_ids=slide_ids)

    predictions = trainer.predict()
    print('model predictions finished')

    print(f'predictions: {predictions}')


if __name__ == "__main__":
    args = parse_agrs()
    predict(args)
    print('Finished processing!')