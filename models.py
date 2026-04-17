import json
import gc
import os
import torch
import pytorch_lightning as pl
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
import evaluate

from modules.loss import LanguageModelCriterion
from modules.metrics import REG_Evaluator, compute_coco_scores
from modules.report_gen_model import ReportGenModel
from utils.utils import extract_fields, read_json_file

class ReportModel(pl.LightningModule):

    def __init__(self, args, tokenizer):
        super().__init__()
        self.model = ReportGenModel(args, tokenizer)#.to(torch.bfloat16)
        # self.model.tie_weights()
        # self.concept_lambda = args.concept_lambda
        for p in self.model.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()

        self.tokenizer = tokenizer
        self.learning_rate = args.lr
        self.__weight_decay = args.weight_decay
        self.__lr_patience =args.lr_patience


        self.val_rouge = ROUGEScore()
        self.test_rouge = ROUGEScore()
        self.reg_evaluator = REG_Evaluator()

        self.predictions = {}

        reports = read_json_file(args.reports_json_path)
        self.reports = {
            report['id'].split('.')[0]: report['report']  for split in reports for report in reports[split]
        }
        # self.reports = {
        #     report['id']: report['report'] for split in reports for report in reports[split]
        # }
        # torch.cuda.set_device(self.trainer.local_rank)


    def get_attn_regularization(self, attns, lambda_entropy=1e-3, lambda_balance=5e-2):
        # Attention Regularization
        _, attn_img, attn_con = attns
        # Mean over layers and heads

        attn_con_mean = attn_con.mean(dim=(0, 1, 2))  # (seq_len, num_concepts)
        attn_img_mean = attn_img.mean(dim=(0, 1, 2))
        # (a) Sparsity regularization (entropy)
        entropy = - (attn_con_mean * torch.log(attn_con_mean + 1e-8)).sum(-1).mean()

        # (b) Balance regularization
        balance = (attn_img_mean.mean() - attn_con_mean.mean()).abs()

        return lambda_entropy * entropy + lambda_balance * balance

    def loss_fn(self, output, reports_ids, reports_masks, attns):
        language_criterion = LanguageModelCriterion()
        caption_loss = language_criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        # concept_loss = self.model.concept_supervision_head(concept_tokens, gecko_concepts)
        # attn_reg = self.get_attn_regularization(attns)
        #
        # with torch.no_grad():
        #     caption_magnitude = caption_loss.detach()
        #     concept_magnitude = concept_loss.detach() + 1e-8
        #     scale = (caption_magnitude / concept_magnitude)
        # concept_loss *= scale
        total_loss = caption_loss #  + self.concept_lambda * concept_loss * scale + attn_reg
        return total_loss


    def training_step(self, batch, batch_idx):
        # print(f'train ----------> {features}')
        gc.collect()
        _, features, report_ids, report_masks = batch
        # print(f'train features: {features}')
        output,attn = self.model(features, report_ids, mode='train')

        loss = self.loss_fn(output, report_ids, report_masks, attn)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        # if batch_idx %1000==0:
        #     print(
        #         f"[GPU] Alloc: {torch.cuda.memory_allocated() / 1e6:.1f} MB | Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        del output
        return loss

    def validation_step(self, batch, batch_idx):
        slide_ids, features, report_ids, report_masks = batch
        # print(f'val features: {features}')
        with torch.no_grad():
            output_,attn = self.model(features, report_ids, mode='train')

            print()

            loss = self.loss_fn(output_, report_ids, report_masks, attn)
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            # self.log('val_c_loss', concept_loss, on_epoch=True, prog_bar=True, sync_dist=True)
            del output_
            torch.cuda.empty_cache()

        if batch_idx % 1==0:
            with torch.no_grad():
                output, attn = self.model(features, report_ids, mode='sample')
                # self.__visualize_attn(attn)
                output = output.detach().cpu().numpy()
                pred_texts = self.tokenizer.batch_decode(output)
                target_texts = [self.reports[slide_id] for slide_id in slide_ids]
                ground_truths = self.tokenizer.batch_decode(report_ids[:, 1:].cpu().numpy())
                self.__save_predictions(slide_ids, pred_texts, ground_truths)
                # self.__print_results(slide_ids, pred_texts, ground_truths)
                rouge_score = self.val_rouge(pred_texts, target_texts)['rouge1_fmeasure'].to(self.device)
                self.log('val_rouge', rouge_score, on_epoch=True, prog_bar=True, sync_dist=True)



    def test_step(self, batch, batch_idx):
        slide_ids, features, report_ids, report_masks = batch

        with torch.no_grad():
            output_,attn  = self.model(features, report_ids, mode='train')
            loss = self.loss_fn(output_, report_ids, report_masks, attn)
            self.log('test_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            del output_
            torch.cuda.empty_cache()

        with torch.no_grad():
            output,attn = self.model(features, report_ids, mode='sample')
            output = output.detach().cpu().numpy()
            self.__visualize_attn(attn)
            pred_texts = self.tokenizer.batch_decode(output)
            target_texts = [self.reports[slide_id] for slide_id in slide_ids]
            ground_truths = self.tokenizer.batch_decode(report_ids[:, 1:].cpu().numpy())
            self.__save_predictions(slide_ids, pred_texts, ground_truths)
            if batch_idx % 10 == 0:
                self.__print_results(slide_ids, pred_texts, ground_truths)

            rouge_score = self.test_rouge(pred_texts, target_texts)['rouge1_fmeasure'].to(self.device)
            self.log('test_rouge', rouge_score, on_epoch=True, prog_bar=True, sync_dist=True)



    def predict_step(self, batch):
        slide_ids, features = batch
        with torch.no_grad():
            output,concept_attn_maps = self.model(features, mode='sample')
        pred_texts = self.tokenizer.batch_decode(output.detach().cpu().numpy())
        target_texts = [self.reports[slide_id] for slide_id in slide_ids]

        self.__print_results(slide_ids[0], pred_texts[0], target_texts[0])

        del output
        return slide_ids,pred_texts

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        self.__log_reg_metrics('val', 'reg', self.reg_evaluator.get_metrics, False)
        self.__log_reg_metrics('val', 'coco', compute_coco_scores, True)
        self.predictions.clear()

    def on_test_epoch_end(self):
        torch.cuda.empty_cache()
        self.__log_reg_metrics('test', 'reg', self.reg_evaluator.get_metrics, False)
        self.__log_reg_metrics('test', 'coco', compute_coco_scores, True)
        self.predictions.clear()

    def configure_optimizers(self):
        d_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(d_params, lr=self.learning_rate, weight_decay=self.__weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.__lr_patience)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def __print_results(self, slide_ids, pred_texts, target_texts):
        RED = '\033[91m'
        RESET = '\033[0m'
        BLUE = '\033[94m'

        for i in range(len(slide_ids)):
            ground_truth = self.reports[slide_ids[i]]

            print('*' * 100)
            print(f'{RED} Predicted report for slide: {slide_ids[i]}: {pred_texts[i]} {RESET}')
            print(f'Ground truth: {target_texts}')
            print(f'{BLUE} Ground truth: {ground_truth} {RESET}')

            # json_string = json.dumps(extract_fields(pred_text), indent=4)
            # print(f'{RED} {json_string} {RESET}')
            print('*' * 100)

    def __calculate_evaluate_metrics(self, pred_texts, target_texts):
        pred_texts = list(map(lambda x: 'placeholder' if x.strip()=='' else x, pred_texts))
        for metric in self.evaluate_metric_scores:
            self.evaluate_metric_scores[metric].append(
                self.evaluate_metrics[metric](pred_texts, target_texts)
            )

    def __save_predictions(self, slide_ids, pred_texts, ground_truths):
        # print(f'slide_ids: {slide_ids}, pred_texts: {pred_texts}')
        for i, slide_id in enumerate(slide_ids):
            self.predictions[slide_id] = {
                'pred': pred_texts[i],
                'target': ground_truths[i]
            }


    def __log_reg_metrics(self, stage, metric_type, evaluate_fn, prog_bar):
        pred_texts = []
        target_texts = []
        for slide_id in self.predictions:
            pred_texts.append(self.predictions[slide_id]['pred'])
            target_texts.append(self.predictions[slide_id]['target'])

        metrics = evaluate_fn(list(zip(pred_texts, target_texts)))

        for metric_name, metric_score in metrics.items():
            self.log(
                f'{stage}_{metric_type}_{metric_name}', metric_score, on_epoch=True, prog_bar=prog_bar, sync_dist=True
            )

    def __visualize_attn(self, weights):
        weights = weights.detach().cpu()
        w_patch = weights[:, :, :, 0].mean().numpy()
        w_slide = weights[:, :, :, 1].mean().numpy()
        w_concept = weights[:, :, :, 2].mean().numpy()

        print(f'w_patch: {w_patch}, w_slide: {w_slide}, w_concept: {w_concept}')