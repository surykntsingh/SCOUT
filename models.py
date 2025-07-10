import torch
import pytorch_lightning as pl
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
import evaluate

from modules.loss import LanguageModelCriterion
from modules.report_gen_model import ReportGenModel


class ReportModel(pl.LightningModule):

    def __init__(self, args, tokenizer, weight_decay=0.01):
        super().__init__()
        self.model = ReportGenModel(args, tokenizer)
        self.tokenizer = tokenizer
        self.__lr = args.lr
        self.__weight_decay = weight_decay
        self.val_rouge = ROUGEScore()
        self.test_bleu = BLEUScore(n_gram=1)
        self.test_rouge = ROUGEScore()
        self.test_bleu = BLEUScore(n_gram=1)
        # self.bleu_2 = BLEUScore(n_gram=2)
        # self.bleu_3 = BLEUScore(n_gram=3)
        # self.bleu_4 = BLEUScore(n_gram=4)
        self.val_meteor = evaluate.load("meteor")
        self.test_meteor = evaluate.load("meteor")
        self.meteor_scores = []

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
        _, patch_feats, pos_feats, report_ids, report_masks, patch_masks = batch

        output_ = self.model(patch_feats, pos_feats, report_ids, patch_masks, mode='train')
        # print(f'val output: {output_}')
        loss = self.loss_fn(output_, report_ids, report_masks)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx % 20 == 0:
            output = self.model(patch_feats, pos_feats, report_ids, patch_masks, mode='sample')
            pred_texts = self.tokenizer.batch_decode(output.cpu().numpy())
            target_texts = self.tokenizer.batch_decode(report_ids[:, 1:].cpu().numpy())
            # print(f'val output: {output_}')
            # print(f'report_ids: {report_ids}, output: {output}')
            print(f'pred_texts: {pred_texts}, target_texts: {target_texts}')

            rouge_score = self.val_rouge(pred_texts, target_texts)
            bleu_score1 = self.val_bleu(pred_texts, target_texts)
            # bleu_score2 = self.bleu_2(pred_texts, target_texts)
            # bleu_score3 = self.bleu_3(pred_texts, target_texts)
            # bleu_score4 = self.bleu_4(pred_texts, target_texts)
            self.meteor_scores.append(
                self.val_meteor.compute(predictions=pred_texts, references=target_texts)['meteor'])
            # print(rouge_score)

            self.log('val_rouge', rouge_score['rouge1_fmeasure'], on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_bleu', bleu_score1, on_epoch=True, prog_bar=True, sync_dist=True)
            # self.log('val_bleu2', bleu_score2, on_epoch=True, prog_bar=True, sync_dist=True)
            # self.log('val_bleu3', bleu_score3, on_epoch=True, prog_bar=True, sync_dist=True)
            # self.log('val_bleu4', bleu_score4, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        _, patch_feats, pos_feats, report_ids, report_masks, patch_masks = batch

        output_ = self.model(patch_feats, pos_feats, report_ids, patch_masks, mode='train')
        loss = self.loss_fn(output_, report_ids, report_masks)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        output = self.model(patch_feats, pos_feats, report_ids, patch_masks, mode='sample')
        pred_texts = self.tokenizer.batch_decode(output.cpu().numpy())
        target_texts = self.tokenizer.batch_decode(report_ids[:, 1:].cpu().numpy())
        print(f'pred_texts: {pred_texts},\n target_texts: {target_texts}')

        rouge_score = self.test_rouge(pred_texts, target_texts)
        bleu_score1 = self.test_bleu(pred_texts, target_texts)
        self.meteor_scores.append(self.test_meteor.compute(predictions=pred_texts, references=target_texts)['meteor'])
        self.log('val_rouge', rouge_score['rouge1_fmeasure'], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_bleu', bleu_score1, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(self, batch):
        _, patch_feats, pos_feats, report_ids, report_masks, patch_masks = batch
        output = self.model(patch_feats, pos_feats, report_ids, patch_masks, mode='sample')
        pred_texts = self.tokenizer.batch_decode(output.cpu().numpy())
        return pred_texts

    def on_validation_epoch_end(self):
        # print(self.meteor_scores)
        meteor_score = sum(self.meteor_scores) / len(self.meteor_scores)
        self.log('val_meteor', meteor_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.meteor_scores.clear()

    def on_test_epoch_end(self):
        # print(self.meteor_scores)
        meteor_score = sum(self.meteor_scores) / len(self.meteor_scores)
        self.log('test_meteor', meteor_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.meteor_scores.clear()

    def configure_optimizers(self):
        d_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(d_params, lr=self.__lr, weight_decay=self.__weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}