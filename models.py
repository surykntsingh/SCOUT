import json
import gc
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
        self.model = ReportGenModel(args, tokenizer)#.to(torch.bfloat16)
        # self.model.tie_weights()

        for p in self.model.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()

        self.tokenizer = tokenizer
        self.learning_rate = args.lr
        self.__weight_decay = args.weight_decay
        self.__lr_patience =args.lr_patience
        self.val_rouge = ROUGEScore()
        self.val_bleu = BLEUScore(n_gram=4)
        self.test_rouge = ROUGEScore()
        self.test_bleu = BLEUScore(n_gram=4)
        # self.bleu_2 = BLEUScore(n_gram=2)
        # self.bleu_3 = BLEUScore(n_gram=3)
        # self.bleu_4 = BLEUScore(n_gram=4)
        self.val_meteor = evaluate.load("meteor")
        self.test_meteor = evaluate.load("meteor")
        self.meteor_scores = []
        self.reg_evaluator = REG_Evaluator()

        self.more_metrics = {
            'emb_score': [],
            'key_score': [],
            'bleu_score': [],
            'rouge_score': [],
            'weighted_score': []
        }
        reports = read_json_file(args.reports_json_path)
        self.reports = {report['id'].split('.')[0]: report['report'] for report in reports}
        # torch.cuda.set_device(self.trainer.local_rank)

        # print(f'self.reports: {self.reports.keys()}')

    def loss_fn(self, output, reports_ids, reports_masks):
        criterion = LanguageModelCriterion()
        loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        return loss

    # def on_after_backward(self):
    #     for name, p in self.model.named_parameters():
    #         if not p.data.is_contiguous():
    #             print(f"NON-CONTIGUOUS PARAM: {name} shape={tuple(p.shape)} strides={p.data.stride()}")

    def training_step(self, batch, batch_idx):
        # print('train ---------->')
        gc.collect()
        _, feats1, feats2, gecko_feats, gecko_concepts, report_ids, report_masks, patch_masks = batch
        output = self.model(feats1, feats2, gecko_feats, gecko_concepts, report_ids, patch_masks, mode='train')
        # print(f'train output: {output}')
        loss = self.loss_fn(output, report_ids, report_masks)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        # if batch_idx %1000==0:
        #     print(
        #         f"[GPU] Alloc: {torch.cuda.memory_allocated() / 1e6:.1f} MB | Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        del output
        return loss

    def validation_step(self, batch, batch_idx):
        slide_ids, feats1, feats2, gecko_feats, gecko_concepts, report_ids, report_masks, patch_masks = batch

        # --- 1. Compute loss safely (no grad tracking) ---
        with torch.no_grad():
            output_ = self.model(
                feats1, feats2, gecko_feats, gecko_concepts, report_ids, patch_masks, mode='train'
            )
            loss = self.loss_fn(output_, report_ids, report_masks)
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        del output_
        torch.cuda.empty_cache()

        # --- 2. Sample predictions periodically ---
        if batch_idx % 50 == 0:
            with torch.no_grad():
                output = self.model(feats1, feats2, gecko_feats, gecko_concepts, report_ids, patch_masks, mode='sample')

                # Detach + move to CPU immediately
                pred_texts = self.tokenizer.batch_decode(output.detach().cpu().numpy())
                del output
                torch.cuda.empty_cache()

                # Get ground truth reports (CPU only)
                target_texts = [self.reports[slide_id] for slide_id in slide_ids]

                # --- 3. Compute metrics safely on CPU ---
                rouge_result = self.val_rouge(pred_texts, target_texts)
                rouge_score = float(rouge_result['rouge1_fmeasure'].cpu().item())

                meteor_result = self.val_meteor.compute(predictions=pred_texts, references=target_texts)
                meteor_score = float(meteor_result['meteor'])

                # Immediate logging; no accumulation in lists
                self.log('val_rouge', rouge_score, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('val_meteor', meteor_score, on_epoch=True, prog_bar=True, sync_dist=True)

        # Free everything ASAP
        del feats1, feats2, gecko_feats, gecko_concepts, report_ids, report_masks, patch_masks
        gc.collect()
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        slide_ids, feats1, feats2, gecko_feats, gecko_concepts, report_ids, report_masks, patch_masks = batch

        # --- 1. Compute loss safely ---
        with torch.no_grad():
            output_ = self.model(
                feats1, feats2, gecko_feats, gecko_concepts, report_ids, patch_masks, mode='train'
            )
            loss = self.loss_fn(output_, report_ids, report_masks)
            self.log('test_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            del output_

        torch.cuda.empty_cache()

        # --- 2. Sample and print every 100 batches ---
        if batch_idx % 1 == 0:
            with torch.no_grad():
                output = self.model(
                    feats1, feats2, gecko_feats, gecko_concepts, report_ids, patch_masks, mode='sample'
                )
                pred_texts = self.tokenizer.batch_decode(output.detach().cpu().numpy())
                del output
                torch.cuda.empty_cache()

                target_texts = [self.reports[slide_id] for slide_id in slide_ids]

                print('*' * 100)
                print(f'Predicted report: {pred_texts[0]}')
                print(f'Ground truth: {target_texts[0]}')
                print('*' * 100)

                # --- 3. Metrics on CPU ---
                rouge_result = self.test_rouge(pred_texts, target_texts)
                rouge_score = float(rouge_result['rouge1_fmeasure'].cpu().item())

                meteor_result = self.test_meteor.compute(predictions=pred_texts, references=target_texts)
                meteor_score = float(meteor_result['meteor'])

                self.log('test_rouge', rouge_score, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('test_meteor', meteor_score, on_epoch=True, prog_bar=True, sync_dist=True)

        # --- 4. Cleanup ---
        del feats1, feats2, gecko_feats, gecko_concepts, report_ids, report_masks, patch_masks
        gc.collect()
        torch.cuda.empty_cache()

    def predict_step(self, batch):
        slide_id, feats1, feats2, gecko_feats, gecko_concepts = batch
        with torch.no_grad():
            output = self.model(feats1, feats2, gecko_feats, gecko_concepts, mode='sample')
        pred_texts = self.tokenizer.batch_decode(output.detach().cpu().numpy())

        RED = '\033[91m'
        RESET = '\033[0m'

        print('*' * 100)
        print(f'{RESET} Predicted report for slide: {slide_id[0]}: {pred_texts[0]} {RESET}')
        print(f' {RED} Predicted synoptic report for slide: {slide_id[0]}: \n {RESET}')

        json_string = json.dumps(extract_fields(pred_texts[0]), indent=4)
        print(f'{RED} {json_string} {RESET}')
        print('*' * 100)
        del output
        return slide_id,pred_texts

    def on_validation_epoch_end(self):
        # print('on_validation_epoch_end start')
        # print(f'meteor_scores: {self.meteor_scores}')
        torch.cuda.empty_cache()
        meteor_score = sum(self.meteor_scores) / len(self.meteor_scores)
        self.log('val_meteor', meteor_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.meteor_scores.clear()

        for metric in self.more_metrics:
            metric_score = sum(self.more_metrics[metric]) / len(self.more_metrics[metric])
            self.log(f'val_{metric}', metric_score, on_epoch=True, prog_bar=True, sync_dist=True)
            self.more_metrics[metric].clear()


    def on_test_epoch_end(self):
        # print(self.meteor_scores)
        meteor_score = sum(self.meteor_scores) / len(self.meteor_scores)
        self.log('test_meteor', meteor_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.meteor_scores.clear()

        for metric in self.more_metrics:
            metric_score = sum(self.more_metrics[metric]) / len(self.more_metrics[metric])
            self.log(f'test_{metric}', metric_score, on_epoch=True, prog_bar=True, sync_dist=True)
            self.more_metrics[metric].clear()

    def configure_optimizers(self):
        d_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(d_params, lr=self.learning_rate, weight_decay=self.__weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.__lr_patience)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}