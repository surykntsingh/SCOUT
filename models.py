import json
import gc
import os
import math
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image, ImageFilter
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
import evaluate

from modules.loss import LanguageModelCriterion
from modules.metrics import REG_Evaluator, compute_coco_scores
from modules.report_gen_model import ReportGenModel
from utils.utils import extract_fields, read_json_file, write_json_file


class ReportModel(pl.LightningModule):

    def __init__(self, args, tokenizer):
        super().__init__()
        self.model = ReportGenModel(args, tokenizer)#.to(torch.bfloat16)
        # self.model.tie_weights()
        self.concept_lambda = args.concept_lambda
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
        # self.reports = {
        #     report['id'].split('.')[0]: report['report']  for split in reports for report in reports[split]
        # }
        self.reports = {
            report['id']: report['report'] for split in reports for report in reports[split]
        }
        # torch.cuda.set_device(self.trainer.local_rank)
        self.__output_dir = getattr(args, 'output_dir', None) or getattr(args, 'results_path', 'results')
        self.__results_dir = getattr(args, 'results_path', self.__output_dir)


    def get_attn_regularization(self, weights, eps=1e-8):
        # Attention Regularization

        if isinstance(weights, dict):
            weights = weights.get("fusion", weights)
        if weights.dim() == 6:
            weights = weights.mean(dim=-1)
        if weights.dim() == 5:
            weights = weights.mean(dim=0)
        entropy = - (weights * (weights + eps).log()).sum(dim=-1)  # [B, L, H]
        return entropy.mean()

    def loss_fn(self, output, reports_ids, reports_masks, attns):
        language_criterion = LanguageModelCriterion()
        caption_loss = language_criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        # concept_loss = self.model.concept_supervision_head(concept_tokens, gecko_concepts)
        attn_reg = self.get_attn_regularization(attns)
        #
        # with torch.no_grad():
        #     caption_magnitude = caption_loss.detach()
        #     concept_magnitude = concept_loss.detach() + 1e-8
        #     scale = (caption_magnitude / concept_magnitude)
        # concept_loss *= scale
        total_loss = caption_loss + self.concept_lambda * attn_reg
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

            loss = self.loss_fn(output_, report_ids, report_masks, attn)
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            # self.log('val_c_loss', concept_loss, on_epoch=True, prog_bar=True, sync_dist=True)
            del output_
            torch.cuda.empty_cache()

        if batch_idx % 10==0:
            with torch.no_grad():
                output, attn = self.model(features, report_ids, mode='sample')
                self.__visualize_attn(attn)
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
        self.__write_predictions()
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

    def __write_predictions(self):
        os.makedirs(self.__results_dir, exist_ok=True)
        predictions = [
            {
                'id': slide_id,
                'predicted': prediction['pred'],
                'ground_trurth': prediction['target']
            }
            for slide_id, prediction in self.predictions.items()
        ]
        write_json_file(predictions, f'{self.__results_dir}/predictions.json')


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
        if isinstance(weights, dict):
            fusion = weights.get("fusion")
            if fusion is None:
                return
            fusion = fusion.detach().cpu()
            if fusion.dim() == 6:
                fusion = fusion.mean(dim=-1)
            if fusion.dim() == 5:
                fusion = fusion.mean(dim=0)
            if fusion.dim() >= 3:
                fusion = fusion.mean(dim=(0, 1, 2))
            print(
                f"fusion weights summary - patch: {fusion[..., 0].mean().item():.4f}, "
                f"slide: {fusion[..., 1].mean().item():.4f}, "
                f"concept: {fusion[..., 2].mean().item():.4f}"
            )
            return

        weights = weights.detach().cpu()
        if weights.dim() == 6:
            weights = weights.mean(dim=-1)
        if weights.dim() == 5:
            weights = weights.mean(dim=0)
        w_patch = weights[:, :, :, 0].mean().numpy()
        w_slide = weights[:, :, :, 1].mean().numpy()
        w_concept = weights[:, :, :, 2].mean().numpy()

        print(f'w_patch: {w_patch}, w_slide: {w_slide}, w_concept: {w_concept}')

    def predict_single_case_with_heatmaps(self, slide_id, features, report_ids=None, report_masks=None,
                                          output_dir=None, layer_idx=-1):
        """
        Run one case through the model, return the sampled prediction, and save
        modality overlays for the chosen decoder layer.
        """
        self.eval()
        device = next(self.parameters()).device

        if report_ids is not None and report_ids.device != device:
            report_ids = report_ids.to(device)
        if report_masks is not None and report_masks.device != device:
            report_masks = report_masks.to(device)
        for key in ("slide", "patch"):
            if key in features and features[key].device != device:
                features[key] = features[key].to(device)
        if "gecko" in features:
            for key in ("deep", "concept"):
                if key in features["gecko"] and features["gecko"][key].device != device:
                    features["gecko"][key] = features["gecko"][key].to(device)

        with torch.no_grad():
            sample_output, _ = self.model(features, mode='sample')
            pred_text = self.tokenizer.batch_decode(sample_output.detach().cpu().numpy())[0]

            if report_ids is None:
                raise ValueError(
                    "report_ids are required to compute decoder attention maps for heatmaps."
                )

            target_text = self.tokenizer.batch_decode(report_ids[:, 1:].detach().cpu().numpy())[0]
            _, attn_maps = self.model(features, report_ids, mode='train')

        save_root = Path(output_dir or self.__results_dir)
        save_root.mkdir(parents=True, exist_ok=True)
        case_dir = save_root / str(slide_id)
        case_dir.mkdir(parents=True, exist_ok=True)

        thumbnail_path = self._find_thumbnail_path(slide_id)
        overlay_paths = self._save_attention_overlays(
            slide_id=slide_id,
            thumbnail_path=thumbnail_path,
            attn_maps=attn_maps,
            output_dir=case_dir,
            layer_idx=layer_idx,
        )
        gate_chart_path = self._save_gate_contribution_chart(
            slide_id=slide_id,
            attn_maps=attn_maps,
            output_dir=case_dir,
            layer_idx=layer_idx,
        )

        return {
            "slide_id": slide_id,
            "prediction": pred_text,
            "target": target_text,
            "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
            "overlay_paths": {k: str(v) for k, v in overlay_paths.items()},
            "gate_chart_path": str(gate_chart_path) if gate_chart_path else None,
            "attn_summary": self._summarize_attention_maps(attn_maps, layer_idx=layer_idx),
        }

    def _summarize_attention_maps(self, attn_maps, layer_idx=-1):
        summary = {}
        if not isinstance(attn_maps, dict):
            return summary

        for modality in ("patch", "slide", "concept"):
            attn = attn_maps.get(modality)
            if attn is None:
                continue
            attn = attn[layer_idx].detach().cpu()  # [B, H, T, S]
            attn = attn.mean(dim=1).mean(dim=1)  # [B, S]
            summary[modality] = attn.squeeze(0).numpy()

        fusion = attn_maps.get("fusion")
        if fusion is not None:
            fusion = fusion[layer_idx].detach().cpu()
            if fusion.dim() == 5:
                fusion = fusion.mean(dim=-1)
            summary["fusion"] = fusion.mean(dim=(1, 2)).squeeze(0).numpy()

        return summary

    def _save_attention_overlays(self, slide_id, thumbnail_path, attn_maps, output_dir, layer_idx=-1):
        overlays = {}
        if not isinstance(attn_maps, dict):
            return overlays

        for modality in ("patch", "slide", "concept"):
            if modality not in attn_maps or attn_maps[modality] is None:
                continue
            attn = attn_maps[modality][layer_idx].detach().cpu()  # [B, H, T, S]
            if attn.dim() != 4:
                raise ValueError(
                    f"Expected {modality} attention to have shape [B, H, T, S], got {tuple(attn.shape)}."
                )
            attn = attn.mean(dim=1).mean(dim=1).squeeze(0)  # [S]
            overlay_path = output_dir / f"{slide_id}_{modality}_overlay.png"
            coords = self._load_coords_for_modality(slide_id, modality)
            self._render_attention_overlay(thumbnail_path, attn, overlay_path, coords=coords)
            overlays[modality] = overlay_path

        return overlays

    def _save_gate_contribution_chart(self, slide_id, attn_maps, output_dir, layer_idx=-1):
        if not isinstance(attn_maps, dict) or attn_maps.get("fusion") is None:
            return None

        fusion = attn_maps["fusion"][layer_idx].detach().cpu().float()
        if fusion.dim() != 5:
            raise ValueError(
                f"Expected fusion weights to have shape [B, T, H, 3, D_head], got {tuple(fusion.shape)}."
            )

        contributions = fusion.mean(dim=(0, 1, 2, 4)).numpy()
        labels = ["Patch", "Slide", "Concept"]
        colors = ["#d95f02", "#1b9e77", "#7570b3"]

        fig = Figure(figsize=(5.0, 3.2), dpi=160)
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        bars = ax.bar(labels, contributions, color=colors, width=0.6)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean gate contribution")
        ax.set_title(f"{slide_id} modality gates")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, value in zip(bars, contributions):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, 0.98),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        chart_path = output_dir / f"{slide_id}_gate_contributions.png"
        fig.savefig(chart_path)
        return chart_path

    def _render_attention_overlay(self, thumbnail_path, scores, output_path, coords=None):
        if thumbnail_path is None:
            raise FileNotFoundError(
                "Could not locate a thumbnail for overlay rendering."
            )

        base_image = Image.open(thumbnail_path).convert("RGB")
        heatmap = self._attention_to_heatmap(scores, base_image.size, coords=coords)
        blended = Image.blend(base_image, heatmap, alpha=0.45)
        blended.save(output_path)

    def _attention_to_heatmap(self, scores, size, coords=None):
        scores = scores.detach().cpu().float().numpy().reshape(-1)
        if scores.size == 0:
            return Image.new("RGB", size, color=(0, 0, 0))

        if coords is not None and len(coords) == scores.size:
            coords = np.asarray(coords)
            if coords.ndim >= 2 and coords.shape[1] >= 2:
                canvas = np.zeros((size[1], size[0]), dtype=np.float32)
                x = coords[:, 0].astype(np.float32)
                y = coords[:, 1].astype(np.float32)
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
                y = (y - y.min()) / (y.max() - y.min() + 1e-8)
                xi = np.clip((x * (size[0] - 1)).round().astype(np.int32), 0, size[0] - 1)
                yi = np.clip((y * (size[1] - 1)).round().astype(np.int32), 0, size[1] - 1)
                np.maximum.at(canvas, (yi, xi), scores)
                canvas = canvas - canvas.min()
                denom = canvas.max()
                if denom > 0:
                    canvas = canvas / denom
                colored = cm.get_cmap("jet")(canvas)[..., :3]
                colored = (colored * 255).astype(np.uint8)
                heatmap = Image.fromarray(colored)
                return heatmap.filter(ImageFilter.GaussianBlur(radius=6))

        grid = int(math.ceil(math.sqrt(scores.size)))
        padded = np.zeros(grid * grid, dtype=np.float32)
        padded[:scores.size] = scores
        padded = padded.reshape(grid, grid)
        padded = padded - padded.min()
        denom = padded.max()
        if denom > 0:
            padded = padded / denom

        colored = cm.get_cmap("jet")(padded)[..., :3]
        colored = (colored * 255).astype(np.uint8)
        heatmap = Image.fromarray(colored).resize(size, resample=Image.BILINEAR)
        return heatmap

    def _load_coords_for_modality(self, slide_id, modality):
        args = self.model.encoder_decoder.args
        path_map = {
            "patch": getattr(args, "embeddings_path_2", None),
            "slide": getattr(args, "embeddings_path", None),
            "concept": getattr(args, "gecko_emb_path", None),
        }
        data_dir = path_map.get(modality)
        if not data_dir:
            return None

        h5_path = Path(data_dir) / f"{slide_id}.h5"
        if not h5_path.exists():
            return None

        try:
            import h5py
        except Exception:
            return None

        coord_keys = ("coords", "patch_coords", "bag_coords")
        with h5py.File(h5_path, "r") as h5_file:
            for key in coord_keys:
                if key in h5_file:
                    coords = h5_file[key][:]
                    return coords
        return None

    def _find_thumbnail_path(self, slide_id):
        base_id = str(slide_id)
        id_variants = {
            base_id,
            base_id.split(".")[0],
            base_id[:12] if len(base_id) > 12 else base_id,
        }
        candidates = [
            f"{variant}.png"
            for variant in id_variants
        ] + [
            f"{variant}.jpg"
            for variant in id_variants
        ] + [
            f"{variant}.jpeg"
            for variant in id_variants
        ]
        search_roots = [Path.cwd(), Path(self.__output_dir).parent, Path(self.__output_dir)]

        for root in search_roots:
            if not root.exists():
                continue
            for candidate in candidates:
                candidate_path = root / candidate
                if candidate_path.exists():
                    return candidate_path

        for root in search_roots:
            if not root.exists():
                continue
            for candidate in candidates:
                matches = list(root.rglob(candidate))
                if matches:
                    return matches[0]

        return None
