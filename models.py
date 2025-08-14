import json

import torch
import pytorch_lightning as pl
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
import evaluate

from modules.loss import LanguageModelCriterion
from modules.metrics import REG_Evaluator
from modules.report_gen_model import ReportGenModel
from utils.utils import extract_fields, read_json_file


class ReportModel(pl.LightningModule):

    def __init__(self, args, tokenizer):
        super().__init__()
        self.model = ReportGenModel(args, tokenizer)
        self.tokenizer = tokenizer
        self.__lr = args.lr
        self.__weight_decay = args.weight_decay
        self.__lr_patience =args.lr_patience
        self.val_rouge = ROUGEScore()
        self.val_bleu = BLEUScore(n_gram=1)
        self.test_rouge = ROUGEScore()
        self.test_bleu = BLEUScore(n_gram=1)
        # self.bleu_2 = BLEUScore(n_gram=2)
        # self.bleu_3 = BLEUScore(n_gram=3)
        # self.bleu_4 = BLEUScore(n_gram=4)
        self.val_meteor = evaluate.load("meteor")
        self.test_meteor = evaluate.load("meteor")
        self.meteor_scores = []
        self.reg_evaluator = REG_Evaluator()
        self.reg_scores = []
        reports = read_json_file(args.reports_json_path)
        self.reports = {report['id'].split('.')[0]: report['report'] for report in reports}

        # print(f'self.reports: {self.reports.keys()}')

    def loss_fn(self, output, reports_ids, reports_masks):
        criterion = LanguageModelCriterion()
        loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        return loss

    def training_step(self, batch):
        # print('train ---------->')
        _, patch_feats, pos_feats, report_ids, report_masks, patch_masks = batch
        output = self.model(patch_feats, pos_feats, report_ids, patch_masks, mode='train')
        # print(f'train output: {output}')
        loss = self.loss_fn(output, report_ids, report_masks)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # print('val ---------->')

        slide_ids, patch_feats, pos_feats, report_ids, report_masks, patch_masks = batch
        # print(
        #     f"[RANK {self.global_rank}] image_feats: {patch_feats.device}, model: {next(self.parameters()).device}")

        output_ = self.model(patch_feats, pos_feats, report_ids, patch_masks, mode='train')
        # print(f'val output: {output_}')
        loss = self.loss_fn(output_, report_ids, report_masks)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx % 20==0:
            output = self.model(patch_feats, pos_feats, report_ids, patch_masks, mode='sample')
            pred_texts = self.tokenizer.batch_decode(output.cpu().numpy())
            # target_texts = self.tokenizer.batch_decode(report_ids[:, 1:].cpu().numpy())

            target_texts = [self.reports[slide_id] for slide_id in slide_ids]

            rouge_score = self.val_rouge(pred_texts, target_texts)['rouge1_fmeasure'].to(self.device)
            bleu_score1 = self.val_bleu(pred_texts, target_texts).to(self.device)
            # bleu_score2 = self.bleu_2(pred_texts, target_texts)
            # bleu_score3 = self.bleu_3(pred_texts, target_texts)
            # bleu_score4 = self.bleu_4(pred_texts, target_texts)
            self.meteor_scores.append(
                self.val_meteor.compute(predictions=pred_texts, references=target_texts)['meteor'])
            self.reg_scores.append(self.reg_evaluator.evaluate_dummy(list(zip(pred_texts, target_texts))))

            self.log('val_rouge', rouge_score, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_bleu', bleu_score1, on_epoch=True, prog_bar=True, sync_dist=True)
            # self.log('val_bleu2', bleu_score2, on_epoch=True, prog_bar=True, sync_dist=True)
            # self.log('val_bleu3', bleu_score3, on_epoch=True, prog_bar=True, sync_dist=True)
            # self.log('val_bleu4', bleu_score4, on_epoch=True, prog_bar=True, sync_dist=True)
            # print('val step end')

    def test_step(self, batch, batch_idx):
        slide_ids, patch_feats, pos_feats, report_ids, report_masks, patch_masks = batch

        output_ = self.model(patch_feats, pos_feats, report_ids, patch_masks, mode='train')
        loss = self.loss_fn(output_, report_ids, report_masks)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        output = self.model(patch_feats, pos_feats, report_ids, patch_masks, mode='sample')
        pred_texts = self.tokenizer.batch_decode(output.cpu().numpy())
        # target_texts = self.tokenizer.batch_decode(report_ids[:, 1:].cpu().numpy())
        # print(f'pred_texts: {pred_texts},\n target_texts: {target_texts}')
        target_texts = [self.reports[slide_id] for slide_id in slide_ids]

        if batch_idx % 100 == 0:
            RED = '\033[91m'
            BLUE = '\033[94m'
            RESET = '\033[0m'

            print('*' * 100)
            print(f'{RESET} Predicted report: {pred_texts[0]} {RESET}')
            print(f' {RED} Predicted synoptic report: \n {RESET}')

            json_string = json.dumps(extract_fields(pred_texts[0]), indent=4)
            print(f'{RED} {json_string} {RESET}')

            print(f'{BLUE} Ground truth: {target_texts[0]} {RESET}')
            print('*' * 100)

        rouge_score = self.val_rouge(pred_texts, target_texts)['rouge1_fmeasure'].to(self.device)
        bleu_score1 = self.val_bleu(pred_texts, target_texts).to(self.device)
        self.meteor_scores.append(self.test_meteor.compute(predictions=pred_texts, references=target_texts)['meteor'])
        self.reg_scores.append(self.reg_evaluator.evaluate_dummy(list(zip(pred_texts, target_texts))))
        self.log('test_rouge', rouge_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_bleu', bleu_score1, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(self, batch):
        slide_id, patch_feats, pos_feats = batch
        output = self.model(patch_feats, pos_feats, mode='sample')
        pred_texts = self.tokenizer.batch_decode(output.cpu().numpy())

        RED = '\033[91m'
        RESET = '\033[0m'

        print('*' * 100)
        print(f'{RESET} Predicted report for slide: {slide_id[0]}: {pred_texts[0]} {RESET}')
        print(f' {RED} Predicted synoptic report for slide: {slide_id[0]}: \n {RESET}')

        json_string = json.dumps(extract_fields(pred_texts[0]), indent=4)
        print(f'{RED} {json_string} {RESET}')
        print('*' * 100)
        return slide_id,pred_texts

    def on_validation_epoch_end(self):
        # print('on_validation_epoch_end start')
        # print(f'meteor_scores: {self.meteor_scores}')
        meteor_score = sum(self.meteor_scores) / len(self.meteor_scores)
        self.log('val_meteor', meteor_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.meteor_scores.clear()

        reg_score = sum(self.reg_scores) / len(self.reg_scores)
        self.log('val_reg', reg_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.reg_scores.clear()

    def on_test_epoch_end(self):
        # print(self.meteor_scores)
        meteor_score = sum(self.meteor_scores) / len(self.meteor_scores)
        self.log('test_meteor', meteor_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.meteor_scores.clear()

        reg_score = sum(self.reg_scores) / len(self.reg_scores)
        self.log('test_reg', reg_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.reg_scores.clear()

    def configure_optimizers(self):
        d_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(d_params, lr=self.__lr, weight_decay=self.__weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.__lr_patience)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}