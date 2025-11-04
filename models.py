import json
import gc
import os
import torch
import pytorch_lightning as pl
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
import evaluate

from modules.loss import LanguageModelCriterion, ConceptSupervisionHead
from modules.metrics import REG_Evaluator, compute_coco_scores
from modules.report_gen_model import ReportGenModel
from utils.utils import extract_fields, read_json_file

class ReportModel(pl.LightningModule):

    def __init__(self, args, tokenizer):
        super().__init__()
        self.model = ReportGenModel(args, tokenizer)#.to(torch.bfloat16)
        # self.model.tie_weights()
        self.concept_lambda = 1
        for p in self.model.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()

        self.tokenizer = tokenizer
        self.learning_rate = args.lr
        self.__weight_decay = args.weight_decay
        self.__lr_patience =args.lr_patience
        self.concept_supervision_head = ConceptSupervisionHead(args.d_model, args.gcd)

        self.val_rouge = ROUGEScore()
        self.val_bleu = BLEUScore(n_gram=4)
        self.test_rouge = ROUGEScore()
        self.test_bleu = BLEUScore(n_gram=4)
        # self.bleu_2 = BLEUScore(n_gram=2)
        # self.bleu_3 = BLEUScore(n_gram=3)
        # self.bleu_4 = BLEUScore(n_gram=4)
        self.val_meteor = evaluate.load("meteor")
        self.test_meteor = evaluate.load("meteor")

        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")
        # bertscore = evaluate.load("bertscore")

        self.evaluate_metrics = {
            'bleu':  lambda x,y: bleu.compute(predictions=x,references=y)['bleu'],
            'rouge': lambda x,y: rouge.compute(predictions=x,references=y)['rougeL'],
            'meteor': lambda x,y: meteor.compute(predictions=x,references=y)['meteor'],
            # 'bertscore': lambda x,y: bertscore.compute(predictions=x,references=y, lang="en")['f1']
        }

        self.evaluate_metric_scores = {
            'bleu': [],
            'rouge': [],
            'meteor': [],
            # 'bertscore': []
        }

        self.reg_evaluator = REG_Evaluator()

        self.reg_metrics = {
            'emb_score': [],
            'key_score': [],
            'bleu_score': [],
            'rouge_score': [],
            'weighted_score': []
        }

        self.coco_metrics = {
            'BLEU_1': [],
            'BLEU_2': [],
            'BLEU_3': [],
            'BLEU_4': [],
            'METEOR': [],
            'ROUGE_L': []
        }
        reports = read_json_file(args.reports_json_path)
        self.reports = {report['id'].split('.')[0]: report['report'] for report in reports}
        # torch.cuda.set_device(self.trainer.local_rank)

        # print(f'self.reports: {self.reports.keys()}')

    def loss_fn(self, output, reports_ids, reports_masks, concept_tokens, gecko_concepts):
        language_criterion = LanguageModelCriterion()
        caption_loss = language_criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        concept_loss = self.concept_supervision_head(concept_tokens, gecko_concepts)

        caption_grad = torch.norm(
            torch.autograd.grad(caption_loss, self.model.decoder.layers[-1].parameters(), retain_graph=True)[0])
        concept_grad = torch.norm(
            torch.autograd.grad(concept_loss, self.model.decoder.layers[-1].parameters(), retain_graph=True)[0])
        scale = (caption_grad / (concept_grad + 1e-6)).clamp(0.1, 10)
        total_loss = caption_loss + self.concept_lambda * scale * concept_loss
        return total_loss



    # def on_after_backward(self):
    #     for name, p in self.model.named_parameters():
    #         if not p.data.is_contiguous():
    #             print(f"NON-CONTIGUOUS PARAM: {name} shape={tuple(p.shape)} strides={p.data.stride()}")

    def training_step(self, batch, batch_idx):
        # print('train ---------->')
        gc.collect()
        _, feats1, feats2, gecko_feats, gecko_concepts, report_ids, report_masks, patch_masks = batch
        output,_, concept_tokens = self.model(feats1, feats2, gecko_feats, gecko_concepts, report_ids, patch_masks, mode='train')
        # print(f'train output: {output}')
        loss = self.loss_fn(output, report_ids, report_masks, concept_tokens,gecko_concepts)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        # if batch_idx %1000==0:
        #     print(
        #         f"[GPU] Alloc: {torch.cuda.memory_allocated() / 1e6:.1f} MB | Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        del output
        return loss

    def validation_step(self, batch, batch_idx):
        # print('val ---------->')
        # gc.collect()
        slide_ids, feats1, feats2, gecko_feats, gecko_concepts, report_ids, report_masks, patch_masks = batch
        # print(
        #     f"[RANK {self.global_rank}] image_feats: {patch_feats.device}, model: {next(self.parameters()).device}")
        with torch.no_grad():
            output_,_,concept_tokens = self.model(feats1, feats2, gecko_feats, gecko_concepts, report_ids, patch_masks, mode='train')

            loss = self.loss_fn(output_, report_ids, report_masks, concept_tokens,gecko_concepts)
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            del output_
            torch.cuda.empty_cache()

        if batch_idx % 50==0:
            with torch.no_grad():
                output, concept_attn_maps_ = self.model(feats1, feats2, gecko_feats, gecko_concepts, report_ids, patch_masks, mode='sample')
                pred_texts = self.tokenizer.batch_decode(output.detach().cpu().numpy())
                # target_texts = self.tokenizer.batch_decode(report_ids[:, 1:].cpu().numpy())
                print(f'concept_attn_maps:: {concept_attn_maps}')
                target_texts = [self.reports[slide_id] for slide_id in slide_ids]

                # gts = {slide_id: [self.reports[slide_id]] for slide_id in slide_ids}
                # preds = {slide_id: [pred_texts[i]] for i,slide_id in enumerate(slide_ids)}

                rouge_score = float(self.val_rouge(pred_texts, target_texts)['rouge1_fmeasure'].to('cpu'))
                bleu_score1 = self.val_bleu(pred_texts, target_texts)
                metrics = self.reg_evaluator.get_metrices(pred_texts, target_texts)
                # coco_metrics = compute_coco_scores(preds, gts)

                for metric in self.evaluate_metric_scores:
                    self.evaluate_metric_scores[metric].append(
                        self.evaluate_metrics[metric](pred_texts, target_texts)
                    )
                # self.reg_scores.append(reg)
                for metric in self.reg_metrics:
                    self.reg_metrics[metric].append(float(metrics[metric]))

                # for metric in self.coco_metrics:
                #     self.coco_metrics[metric].append(float(coco_metrics[metric]))

                self.log('val_rouge', rouge_score, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('val_bleu', bleu_score1, on_epoch=True, prog_bar=True, sync_dist=True)
                del output
                del feats1, feats2, gecko_feats, gecko_concepts, report_ids, report_masks, patch_masks
                gc.collect()
                torch.cuda.empty_cache()


    def test_step(self, batch, batch_idx):
        # gc.collect()
        slide_ids, feats1, feats2, gecko_feats, gecko_concepts, report_ids, report_masks, patch_masks = batch

        with torch.no_grad():
            output_,_,concept_tokens  = self.model(feats1, feats2, gecko_feats, gecko_concepts, report_ids, patch_masks, mode='train')
            loss = self.loss_fn(output_, report_ids, report_masks, concept_tokens, gecko_concepts)
            self.log('test_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            del output_
            torch.cuda.empty_cache()

        with torch.no_grad():
            output,concept_attn_maps, _ = self.model(feats1, feats2, gecko_feats, gecko_concepts, report_ids, patch_masks, mode='sample')
            pred_texts = self.tokenizer.batch_decode(output.detach().cpu().numpy())

            target_texts = [self.reports[slide_id] for slide_id in slide_ids]

            # gts = {slide_id: [self.reports[slide_id]] for slide_id in slide_ids}
            # preds = {slide_id: [pred_texts[i]] for i, slide_id in enumerate(slide_ids)}

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
                print(f'concept_attn_maps:: {concept_attn_maps}')
            rouge_score = float(self.test_rouge(pred_texts, target_texts)['rouge1_fmeasure'].to('cpu'))
            bleu_score1 = self.test_bleu(pred_texts, target_texts).to(self.device)
            # meteor_score = float(self.test_meteor.compute(predictions=pred_texts, references=target_texts)['meteor'])

            for metric in self.evaluate_metric_scores:
                self.evaluate_metric_scores[metric].append(
                    self.evaluate_metrics[metric](pred_texts, target_texts)
                )

            metrics = self.reg_evaluator.get_metrices(pred_texts, target_texts)
            # coco_metrics = compute_coco_scores(gts, preds)
            for metric in self.reg_metrics:
                self.reg_metrics[metric].append(float(metrics[metric]))

            # for metric in self.coco_metrics:
            #     self.coco_metrics[metric].append(float(coco_metrics[metric]))
            self.log('test_rouge', rouge_score, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('test_bleu', bleu_score1, on_epoch=True, prog_bar=True, sync_dist=True)
            del output
            del feats1, feats2, gecko_feats, gecko_concepts, report_ids, report_masks, patch_masks
            gc.collect()
            torch.cuda.empty_cache()


    def predict_step(self, batch):
        slide_ids, feats1, feats2, gecko_feats, gecko_concepts = batch
        with torch.no_grad():
            output = self.model(feats1, feats2, gecko_feats, gecko_concepts, mode='sample')
        pred_texts = self.tokenizer.batch_decode(output.detach().cpu().numpy())
        target_texts = [self.reports[slide_id] for slide_id in slide_ids]
        RED = '\033[91m'
        RESET = '\033[0m'
        BLUE = '\033[94m'

        print('*' * 100)
        print(f'{RESET} Predicted report for slide: {slide_ids[0]}: {pred_texts[0]} {RESET}')
        print(f' {RED} Predicted synoptic report for slide: {slide_ids[0]}: \n {RESET}')

        print(f'{BLUE} Ground truth: {target_texts[0]} {RESET}')

        json_string = json.dumps(extract_fields(pred_texts[0]), indent=4)
        print(f'{RED} {json_string} {RESET}')
        print('*' * 100)
        del output
        return slide_ids,pred_texts

    def on_validation_epoch_end(self):
        # print('on_validation_epoch_end start')
        # print(f'meteor_scores: {self.meteor_scores}')
        torch.cuda.empty_cache()
        # meteor_score = sum(self.evaluate_metric_scores) / len(self.evaluate_metric_scores)
        # self.log('val_meteor', meteor_score, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.evaluate_metric_scores.clear()

        for metric in self.reg_metrics:
            metric_score = sum(self.reg_metrics[metric]) / len(self.reg_metrics[metric])
            self.log(f'val_{metric}', metric_score, on_epoch=True, prog_bar=True, sync_dist=True)
            self.reg_metrics[metric].clear()

        print(self.evaluate_metric_scores)
        for metric in self.evaluate_metric_scores:
            metric_score = sum(self.evaluate_metric_scores[metric]) / len(self.evaluate_metric_scores[metric])
            self.log(f'val_e_{metric}', metric_score, on_epoch=True, prog_bar=True, sync_dist=True)
            self.evaluate_metric_scores[metric].clear()


    def on_test_epoch_end(self):
        # print(self.meteor_scores)
        # meteor_score = sum(self.evaluate_metric_scores) / len(self.evaluate_metric_scores)
        # self.log('test_meteor', meteor_score, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.evaluate_metric_scores.clear()

        for metric in self.reg_metrics:
            metric_score = sum(self.reg_metrics[metric]) / len(self.reg_metrics[metric])
            self.log(f'test_{metric}', metric_score, on_epoch=True, prog_bar=True, sync_dist=True)
            self.reg_metrics[metric].clear()

        for metric in self.evaluate_metric_scores:
            metric_score = sum(self.evaluate_metric_scores[metric]) / len(self.evaluate_metric_scores[metric])
            self.log(f'test_e_{metric}', metric_score, on_epoch=True, prog_bar=True, sync_dist=True)
            self.evaluate_metric_scores[metric].clear()

    def configure_optimizers(self):
        d_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(d_params, lr=self.learning_rate, weight_decay=self.__weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.__lr_patience)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}