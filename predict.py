from models import ReportModel
from report_tokenizers import Tokenizer
from trainer import Trainer
from utils.arg_parser import parse_agrs



def predict(args):

    tokenizer = Tokenizer(args.reports_json_path)
    model_path = args.model_save_path

    model = ReportModel.load_from_checkpoint(model_path, args=args, tokenizer=tokenizer)
    slide_ids = ['PIT_01_04212_01', 'PIT_01_09950_01', 'PIT_03_01569_01', 'PIT_01_08982_01',
                 'PIT_01_01683_01', 'PIT_01_04165_01', 'PIT_03_00832_01', 'PIT_03_01590_01',
                 'PIT_03_01675_01', 'PIT_01_00872_01'
                 ]
    trainer = Trainer(args, model, tokenizer, predict=True, slide_ids=slide_ids)

    predictions = trainer.predict()
    print('model predictions finished')
    results = {}

    # for batch in predictions:
    #     print(f'batch: {batch}')
    print(f'predictions: {predictions}')
    for slide_ids, reports  in predictions:
        for i in range(args.batch_size):
            results[slide_ids[i]] = reports[i]

    print(f'results: {results}')


if __name__ == "__main__":
    args = parse_agrs()
    predict(args)
    print('Finished processing!')