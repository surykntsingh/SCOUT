from pprint import pprint

from models import ReportModel
from report_tokenizers import Tokenizer
from trainer import Trainer
from utils.arg_parser import parse_agrs
from utils.utils import save_results


def predict(args):

    tokenizer = Tokenizer(args.reports_json_path)
    model_path = args.model_save_path

    model = ReportModel.load_from_checkpoint(model_path, args=args, tokenizer=tokenizer)
    # slide_ids = ['PIT_01_04212_01', 'PIT_01_09950_01', 'PIT_03_01569_01', 'PIT_01_08982_01',
    #              'PIT_01_01683_01', 'PIT_01_04165_01', 'PIT_03_00832_01', 'PIT_03_01590_01',
    #              'PIT_03_01675_01', 'PIT_01_00872_01'
    #              ]
    trainer = Trainer(args, model, tokenizer, predict=True)

    predictions = trainer.predict()
    print('model predictions finished')
    results = []

    print(f'predictions: {predictions}')
    for slide_ids, reports  in predictions:
        for i in range(args.batch_size):
            results.append({
                'id': f'{slide_ids[i]}.tiff',
                'report': reports[i]
            })

    # pprint(results)
    return results


if __name__ == "__main__":
    args = parse_agrs()
    results = predict(args)
    print(f'Saving results at path: {args.results_path} ')
    save_results(results, args.results_path)
    print('Finished processing!')