import typer
import os
from datamodules.wsi_embedding_datamodule import PatchEmbeddingDataModule, PatchEmbeddingDataPredictModule
from models import ReportModel
from report_tokenizers import Tokenizer
from trainer import Trainer, KFoldTrainer
import pandas as pd
from utils.utils import save_model, get_params_for_key, copy_yaml, write_json_file
from datetime import datetime
import torch

app = typer.Typer(pretty_exceptions_enable=False)


torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')
torch.autograd.set_detect_anomaly(True)

@app.command()
def train(config_file_path: str='histai_config.yaml', reg_threshold: float=0.8):
    args = get_params_for_key(config_file_path, "train")
    split_frac = [0.85, 0.07, 0.08]
    tokenizer = Tokenizer(args.reports_json_path)
    if args.resume:
        print(f'Resuming training... Loading model from {args.model_load_path}')
        model = ReportModel.load_from_checkpoint(args.model_load_path, args=args, tokenizer=tokenizer)
    else:
        model = ReportModel(args, tokenizer)

    datamodule = PatchEmbeddingDataModule(args, tokenizer, split_frac)
    date = datetime.now()
    args.ckpt_path +=  f'/{date.strftime("%Y%m%d")}/{date.strftime("%H%M%S")}'
    os.makedirs(args.ckpt_path, exist_ok=True)
    trainer = Trainer(args, tokenizer, split_frac)
    train_metrics, _ = trainer.train(model, datamodule, fast_dev_run=args.fast_dev_run)
    print('model training finished')

    best_model_path = trainer.best_model_path
    # if not args.fast_dev_run:
    print(f'loading best model from {best_model_path}')

    best_model = ReportModel.load_from_checkpoint(best_model_path, args=args, tokenizer=tokenizer)
    test_metrics, tr = trainer.test(best_model, datamodule, fast_dev_run=args.fast_dev_run)
    print('model testing finished')
    # save_model(args, trainer)


    metrics = {**train_metrics, **test_metrics, 'best_model_path': best_model_path}
    print(f'train_metrics: {train_metrics}, test_metrics: {test_metrics})') #, tune_metrics: {tune_metrics}')

    copy_yaml(config_file_path, args.ckpt_path)
    os.makedirs(f'{args.ckpt_path}/results', exist_ok=True)
    write_metrics(f'{args.ckpt_path}/results', metrics, date)

    os.makedirs(f'{args.results_path}/experiments', exist_ok=True)
    write_metrics(f'{args.results_path}/experiments', metrics, date)

    # if test_metrics['test_reg'].item() > reg_threshold:
    #     print(f'Generating predictions since reg_score > {reg_threshold}')
    #     results = predict(best_model, trainer, args, tokenizer)
    #     results_dir = f'{args.ckpt_path}/results_{test_metrics["test_reg"]}'
    #     print(f'Saving predictions at {results_dir}')
    #     save_results(results, results_dir)
    #     print(f'Predictions saved at {results_dir}')
        # os.makedirs(f'{args.ckpt_path}/saved_models', exist_ok=True)
        # save_model_path = f'{args.ckpt_path}/saved_models/reg_{test_metrics["test_reg"]}.ckpt'
        # trainer.save_model(tr, save_model_path)
    # else:
    #     print(f'Not generating predictions since reg_score < {reg_threshold}')

    # tune_gecko_features(args, tokenizer, best_model_path, trainer, datamodule, reg_threshold, date)



def tune_gecko_features(args, tokenizer,best_model_path, trainer, datamodule, reg_threshold, date):
    # gecko tuning
    print('tuning on gecko concept features')
    model = ReportModel.load_from_checkpoint(best_model_path, args=args, tokenizer=tokenizer)
    tune_metrics, _ = trainer.tune(model, datamodule, fast_dev_run=args.fast_dev_run)
    print('model tuning finished')
    print(f'tune_metrics: {tune_metrics}')

    best_model_path = trainer.best_model_path
    # if not args.fast_dev_run:
    print(f'loading best model from {best_model_path}')
    best_model = ReportModel.load_from_checkpoint(best_model_path, args=args, tokenizer=tokenizer)
    test_metrics, tr = trainer.test(best_model, datamodule, fast_dev_run=args.fast_dev_run)
    print('model testing finished')
    # save_model(args, trainer)

    metrics = {**tune_metrics, **test_metrics, 'best_model_path': best_model_path}
    print(f'tune_metrics: {tune_metrics}, test_metrics: {test_metrics})')
    write_metrics(f'{args.ckpt_path}/results', metrics, date)
    write_metrics(f'{args.results_path}/experiments', metrics, date)

    if test_metrics['test_reg'].item() > reg_threshold:
        print(f'Generating predictions since reg_score > {reg_threshold}')
        results = predict(best_model, trainer, args, tokenizer)
        results_dir = f'{args.ckpt_path}/results_{test_metrics["test_reg"]}'
        print(f'Saving predictions at {results_dir}')
        save_results(results, results_dir)
        print(f'Predictions saved at {results_dir}')
        # os.makedirs(f'{args.ckpt_path}/saved_models', exist_ok=True)
        # save_model_path = f'{args.ckpt_path}/saved_models/reg_{test_metrics["test_reg"]}.ckpt'
        # trainer.save_model(tr, save_model_path)
    else:
        print(f'Not generating predictions since reg_score < {reg_threshold}')


@app.command()
def test(config_file_path: str='histai_config.yaml', reg_threshold: float=0.8):

    args = get_params_for_key(config_file_path, "train")
    split_frac = [0.85, 0.07, 0.08]
    tokenizer = Tokenizer(args.reports_json_path)
    datamodule = PatchEmbeddingDataModule(args, tokenizer, split_frac)
    trainer = Trainer(args, tokenizer, split_frac)

    print(f'loading best model from {args.model_load_path}')
    model = ReportModel.load_from_checkpoint(args.model_load_path, args=args, tokenizer=tokenizer)
    test_metrics, tr = trainer.test(model, datamodule, fast_dev_run=args.fast_dev_run)
    print(f'test_metrics: {test_metrics}')
    print('model testing finished')
    #
    # if test_metrics['test_reg'].item() > reg_threshold:
    #     print(f'Generating predictions since reg_score > {reg_threshold}')
    #     results = predict(model, trainer, args, tokenizer)
    #     results_dir = f'{args.results_path}/results_{test_metrics["test_reg"]}'
    #     print(f'Saving predictions at {results_dir}')
    #     save_results(results, results_dir)
    # else:
    #     print(f'Not generating predictions since reg_score < {reg_threshold}')


@app.command()
def trainkfold(config_file_path='tcga_config.yaml'):
    args = get_params_for_key(config_file_path, "train")
    split_frac = [0.85, 0.15]
    tokenizer = Tokenizer(args.reports_json_path)
    # model = ReportModel(args, tokenizer)
    date = datetime.now()
    args.ckpt_path +=  f'/{date.strftime("%Y%m%d")}/{date.strftime("%H%M%S")}'

    trainer = KFoldTrainer(args, tokenizer, split_frac)
    trainer.train_and_test(fast_dev_run=args.fast_dev_run)

    metrics = trainer.get_metrics()
    print(f'metrics: {metrics}')
    write_metrics(f'{args.results_path}/experiments', metrics, date)

@app.command()
def predict(config_file_path='histai_config.yaml', pt=False):
    args = get_params_for_key(config_file_path, "train")
    tokenizer = Tokenizer(args.reports_json_path)
    # if pt:
    #     model = ReportModel(args, tokenizer)
    #     model.load_state_dict(torch.load(args.model_load_path))
    # else:
    model = ReportModel.load_from_checkpoint(args.model_load_path, args=args, tokenizer=tokenizer)

    split_frac = [0.85, 0.15]
    trainer = Trainer(args, tokenizer, split_frac)
    results = get_prediction(model, trainer, args, tokenizer)
    results_dir = f'{args.results_path}/predicted_results'
    os.makedirs(results_dir, exist_ok=True)

    save_results(results, results_dir)

def get_prediction(model, trainer, args, tokenizer):
    slides = ['TCGA-VP-A87D',
 'TCGA-J4-A67K',
 'TCGA-G9-6336',
 'TCGA-EJ-7327',
 'TCGA-XJ-A9DI',
 'TCGA-HC-7230',
 'TCGA-XJ-A83H',
 'TCGA-CH-5768',
 'TCGA-HC-7231',
 'TCGA-YL-A9WX',
 'TCGA-HC-8262',
 'TCGA-CH-5764',
 'TCGA-V1-A8MM',
 'TCGA-KK-A7AP',
 'TCGA-G9-7525',
 'TCGA-V1-A9Z7',
 'TCGA-FC-A66V',
 'TCGA-HC-8265',
 'TCGA-EJ-5518',
 'TCGA-G9-6377',
 'TCGA-VP-A876',
 'TCGA-KK-A7B3',
 'TCGA-HC-8212',
 'TCGA-HC-7232',
 'TCGA-CH-5762',
 'TCGA-YL-A8HJ',
 'TCGA-FC-7708',
 'TCGA-G9-6354',
 'TCGA-YL-A8S9',
 'TCGA-HC-7741',
 'TCGA-EJ-5510',
 'TCGA-KK-A6E2',
 'TCGA-HC-8260',
 'TCGA-V1-A8WN',
 'TCGA-XK-AAIR',
 'TCGA-YL-A9WH',
 'TCGA-KK-A7AU',
 'TCGA-KK-A8IJ',
 'TCGA-M7-A722',
 'TCGA-EJ-5516',
 'TCGA-EJ-A46F',
 'TCGA-G9-7522',
 'TCGA-YL-A8SR',
 'TCGA-KK-A7B1',
 'TCGA-HC-7818',
 'TCGA-G9-6367',
 'TCGA-XK-AAJR',
 'TCGA-G9-A9S7',
 'TCGA-SU-A7E7',
 'TCGA-HC-A4ZV',
 'TCGA-KK-A7AZ',
 'TCGA-YL-A8SA',
 'TCGA-V1-A8WW',
 'TCGA-G9-7519',
 'TCGA-EJ-7783',
 'TCGA-HC-8258',
 'TCGA-G9-6333',
 'TCGA-G9-6351',
 'TCGA-HC-7740',
 'TCGA-G9-6348',
 'TCGA-YL-A8SF',
 'TCGA-V1-A8MF',
 'TCGA-KK-A59Z',
 'TCGA-XJ-A9DK',
 'TCGA-V1-A8WL',
 'TCGA-H9-A6BX',
 'TCGA-G9-6369',
 'TCGA-V1-A9O7',
 'TCGA-XK-AAJP',
 'TCGA-M7-A71Z',
 'TCGA-CH-5737',
 'TCGA-2A-A8VO',
 'TCGA-VP-AA1N',
 'TCGA-J4-A67M',
 'TCGA-G9-6353',
 'TCGA-EJ-7789',
 'TCGA-QU-A6IN',
 'TCGA-V1-A9OL',
 'TCGA-J4-A83I',
 'TCGA-HC-A76X',
 'TCGA-YL-A8S8',
 'TCGA-EJ-5532',
 'TCGA-YL-A8SC',
 'TCGA-2A-A8W1',
 'TCGA-KK-A7AW',
 'TCGA-KK-A6E6',
 'TCGA-KK-A59V',
 'TCGA-X4-A8KS',
 'TCGA-V1-A9OT',
 'TCGA-HC-7210',
 'TCGA-EJ-5527',
 'TCGA-YL-A8HL',
 'TCGA-XK-AAJ3',
 'TCGA-HC-8259',
 'TCGA-XK-AAJA',
 'TCGA-EJ-7786',
 'TCGA-HC-7745',
 'TCGA-V1-A8ML',
 'TCGA-YL-A8SI',
 'TCGA-CH-5750',
 'TCGA-G9-7510',
 'TCGA-M7-A723',
 'TCGA-KK-A8I6',
 'TCGA-EJ-A46D',
 'TCGA-V1-A9O5',
 'TCGA-V1-A9ZG',
 'TCGA-4L-AA1F',
 'TCGA-EJ-7314',
 'TCGA-H9-7775',
 'TCGA-XK-AAJU',
 'TCGA-EJ-5526',
 'TCGA-EJ-5499',
 'TCGA-CH-5740',
 'TCGA-EJ-5519',
 'TCGA-2A-A8VX',
 'TCGA-KK-A8I4',
 'TCGA-HC-7078',
 'TCGA-EJ-5505',
 'TCGA-G9-6364',
 'TCGA-HC-8261',
 'TCGA-EJ-5514',
 'TCGA-G9-6496',
 'TCGA-YL-A8SB',
 'TCGA-V1-A8WS',
 'TCGA-KK-A8IG',
 'TCGA-V1-A9OY',
 'TCGA-HC-A9TE',
 'TCGA-CH-5739',
 'TCGA-EJ-5496',
 'TCGA-2A-A8VL',
 'TCGA-EJ-5530',
 'TCGA-EJ-A46G',
 'TCGA-X4-A8KQ',
 'TCGA-HC-7080',
 'TCGA-YL-A9WK',
 'TCGA-VP-A87H',
 'TCGA-XK-AAIW',
 'TCGA-G9-6343',
 'TCGA-M7-A71Y',
 'TCGA-HC-7817',
 'TCGA-KK-A6E8',
 'TCGA-HC-A6HY',
 'TCGA-HC-7820',
 'TCGA-EJ-5517',
 'TCGA-CH-5748',
 'TCGA-V1-A9OX',
 'TCGA-HC-8264',
 'TCGA-2A-AAYF',
 'TCGA-J4-A67O',
 'TCGA-J4-A83J',
 'TCGA-TP-A8TT',
 'TCGA-2A-A8W3',
 'TCGA-YL-A8SL',
 'TCGA-HC-7742',
 'TCGA-V1-A8MU',
 'TCGA-FC-A5OB',
 'TCGA-XQ-A8TB',
 'TCGA-V1-A8MG',
 'TCGA-XJ-A9DX',
 'TCGA-TK-A8OK',
 'TCGA-HC-A6AN',
 'TCGA-KK-A7B2',
 'TCGA-EJ-7788',
 'TCGA-KK-A6E0',
 'TCGA-M7-A725',
 'TCGA-EJ-5497',
 'TCGA-EJ-A46E',
 'TCGA-V1-A8MK',
 'TCGA-2A-A8VT',
 'TCGA-2A-AAYU',
 'TCGA-2A-A8VV',
 'TCGA-2A-AAYO',
 'TCGA-M7-A721',
 'TCGA-YL-A8SO',
 'TCGA-J4-A67T',
 'TCGA-EJ-5524',
 'TCGA-J4-A67N',
 'TCGA-HC-A8D1',
 'TCGA-G9-7521',
 'TCGA-HC-7212',
 'TCGA-M7-A724',
 'TCGA-CH-5766',
 'TCGA-G9-6342',
 'TCGA-HC-7738',
 'TCGA-HC-8213',
 'TCGA-YL-A8SJ',
 'TCGA-HC-7736',
 'TCGA-HC-7079',
 'TCGA-HC-7075',
 'TCGA-V1-A8MJ',
 'TCGA-EJ-7785',
 'TCGA-M7-A720',
 'TCGA-EJ-7784',
 'TCGA-J4-A67Q',
 'TCGA-KK-A7B0',
 'TCGA-HI-7168',
 'TCGA-KK-A8IA',
 'TCGA-HI-7170',
 'TCGA-VP-A879',
 'TCGA-G9-6356',
 'TCGA-YL-A8SP',
 'TCGA-HC-A632',
 'TCGA-EJ-7315',
 'TCGA-EJ-5498',
 'TCGA-G9-6329',
 'TCGA-HC-7209',
 'TCGA-HC-A6AS',
 'TCGA-HC-A6AL',
 'TCGA-WW-A8ZI',
 'TCGA-CH-5767',
 'TCGA-CH-5741',
 'TCGA-KK-A6E1',
 'TCGA-EJ-5531',
 'TCGA-V1-A9O9',
 'TCGA-Y6-A8TL',
 'TCGA-HC-A6HX',
 'TCGA-QU-A6IM',
 'TCGA-G9-A9S4',
 'TCGA-HC-7077',
 'TCGA-CH-5761',
 'TCGA-EJ-7115',
 'TCGA-HC-8256',
 'TCGA-EJ-7330',
 'TCGA-EJ-5512',
 'TCGA-HI-7169',
 'TCGA-HC-A6AO',
 'TCGA-YL-A9WL',
 'TCGA-EJ-5502',
 'TCGA-KK-A8IF',
 'TCGA-KK-A8IC',
 'TCGA-KK-A5A1',
 'TCGA-VP-A872',
 'TCGA-EJ-5522',
 'TCGA-HC-7748',
 'TCGA-KK-A8I8',
 'TCGA-J4-AATV',
 'TCGA-EJ-7321',
 'TCGA-EJ-5542',
 'TCGA-FC-A4JI',
 'TCGA-CH-5744',
 'TCGA-J4-8200',
 'TCGA-HC-7233',
 'TCGA-CH-5769',
 'TCGA-HC-7821',
 'TCGA-CH-5738',
 'TCGA-CH-5746',
 'TCGA-YL-A9WI',
 'TCGA-YL-A8HO',
 'TCGA-EJ-A46H',
 'TCGA-G9-6363',
 'TCGA-QU-A6IL',
 'TCGA-MG-AAMC',
 'TCGA-EJ-7794',
 'TCGA-KK-A7AY',
 'TCGA-HC-7749',
 'TCGA-KK-A8I7',
 'TCGA-KK-A7B4',
 'TCGA-HC-8216',
 'TCGA-G9-6373',
 'TCGA-V1-A9ZK',
 'TCGA-KK-A7AQ',
 'TCGA-VP-A87K',
 'TCGA-HC-A631',
 'TCGA-KK-A8I5',
 'TCGA-CH-5743',
 'TCGA-HC-A6AP',
 'TCGA-HC-8257',
 'TCGA-EJ-7793',
 'TCGA-KK-A6DY',
 'TCGA-V1-A9Z8',
 'TCGA-J4-A67L',
 'TCGA-YL-A8SK',
 'TCGA-HC-A9TH',
 'TCGA-YL-A8HK',
 'TCGA-KK-A8IK',
 'TCGA-QU-A6IO',
 'TCGA-KK-A8I9',
 'TCGA-EJ-5506',
 'TCGA-XJ-A83G',
 'TCGA-HC-7747',
 'TCGA-EJ-5501',
 'TCGA-G9-6347',
 'TCGA-J4-A67S',
 'TCGA-XA-A8JR',
 'TCGA-HC-7744',
 'TCGA-KK-A8IM',
 'TCGA-KK-A6E7',
 'TCGA-HC-7750',
 'TCGA-KK-A59Y',
 'TCGA-FC-A6HD',
 'TCGA-VP-A875',
 'TCGA-G9-6339',
 'TCGA-Y6-A9XI',
 'TCGA-HC-A48F',
 'TCGA-G9-6366',
 'TCGA-XJ-A9DQ',
 'TCGA-KK-A8IH',
 'TCGA-HI-7171',
 'TCGA-G9-6385',
 'TCGA-EJ-7791',
 'TCGA-EJ-7317',
 'TCGA-EJ-5525',
 'TCGA-J4-A83L',
 'TCGA-EJ-5515',
 'TCGA-J4-A6G1',
 'TCGA-CH-5754',
 'TCGA-G9-6371',
 'TCGA-EJ-5511',
 'TCGA-EJ-5507',
 'TCGA-YL-A8SH',
 'TCGA-V1-A9OH',
 'TCGA-FC-7961',
 'TCGA-CH-5751',
 'TCGA-TP-A8TV',
 'TCGA-G9-6378',
 'TCGA-HC-7213',
 'TCGA-KK-A6E5',
 'TCGA-J4-A67R',
 'TCGA-EJ-7782',
 'TCGA-EJ-A46B',
 'TCGA-KK-A8ID',
 'TCGA-HC-A6AQ',
 'TCGA-XK-AAIV',
 'TCGA-KK-A6E3',
 'TCGA-CH-5745',
 'TCGA-KK-A8II',
 'TCGA-HC-7819',
 'TCGA-J4-A83K',
 'TCGA-G9-6370',
 'TCGA-EJ-7792',
 'TCGA-EJ-5509',
 'TCGA-EJ-5508',
 'TCGA-KK-A6E4',
 'TCGA-YL-A8HM',
 'TCGA-J4-AATZ',
 'TCGA-XK-AAK1',
 'TCGA-V1-A9OA',
 'TCGA-HC-7211',
 'TCGA-J4-8198',
 'TCGA-CH-5752',
 'TCGA-V1-A8WV',
 'TCGA-XQ-A8TA',
 'TCGA-G9-6498',
 'TCGA-VP-A878',
 'TCGA-V1-A9OQ',
 'TCGA-EJ-5521',
 'TCGA-EJ-5504',
 'TCGA-J4-A83M',
 'TCGA-G9-6384',
 'TCGA-CH-5763',
 'TCGA-FC-A8O0',
 'TCGA-G9-6499',
 'TCGA-EJ-7331',
 'TCGA-J4-A6M7',
 'TCGA-HC-7737',
 'TCGA-G9-6365',
 'TCGA-CH-5753',
 'TCGA-HC-A76W',
 'TCGA-EJ-7123',
 'TCGA-HC-8266',
 'TCGA-EJ-5495',
 'TCGA-YL-A9WY',
 'TCGA-KK-A59X',
 'TCGA-KK-A8IB',
 'TCGA-QU-A6IP',
 'TCGA-VP-A87B',
 'TCGA-G9-6361',
 'TCGA-XJ-A83F',
 'TCGA-YL-A8SQ',
 'TCGA-VP-A87C',
 'TCGA-EJ-5494',
 'TCGA-G9-6338',
 'TCGA-J4-A6G3',
 'TCGA-G9-7523',
 'TCGA-EJ-5503',
 'TCGA-VP-A87E',
 'TCGA-V1-A9ZR',
 'TCGA-EJ-7328',
 'TCGA-H9-A6BY',
 'TCGA-G9-6332',
 'TCGA-V1-A9Z9',
 'TCGA-EJ-A46I',
 'TCGA-KK-A7AV',
 'TCGA-G9-6379',
 'TCGA-KK-A8IL',
 'TCGA-G9-A9S0',
 'TCGA-YL-A9WJ',
 'TCGA-G9-6362',
 'TCGA-V1-A9OF',
 'TCGA-EJ-7797',
 'TCGA-HC-A8D0',
 'TCGA-HC-7081',
 'TCGA-V1-A8X3',
 'TCGA-G9-6494',
 'TCGA-J4-AAU2',
 'TCGA-VP-A87J',
 'TCGA-HC-7752',
 'TCGA-HC-A8CY',
 'TCGA-CH-5765',
 'TCGA-XK-AAJT',
 'TCGA-G9-7509',
 'TCGA-V1-A9ZI',
 'TCGA-J4-A83N']
    datamodule = PatchEmbeddingDataPredictModule(args, tokenizer, slide_ids=slides)
    predictions = trainer.predict(model, datamodule, fast_dev_run=args.fast_dev_run)
    print('model predictions finished')
    results = []

    print(f'predictions: {predictions}')
    for slide_ids, reports in predictions:
        for i in range(args.batch_size):
            results.append({
                'id': f'{slide_ids[i]}.tiff',
                'report': reports[i]
            })

    return results

def write_metrics(results_path, metrics, date):
    metrics['date'] = date.strftime("%Y-%m-%d %H:%M:%S")
    metrics_df = pd.DataFrame([metrics])


    metrics_df.to_csv(f'{results_path}/results.csv',mode='a')

def save_results(results, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    results_path = f'{results_dir}/predictions.json'
    write_json_file(results, results_path)



# args = parse_agrs()
if __name__ == "__main__":
    app()