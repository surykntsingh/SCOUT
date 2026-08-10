import json
import gc
import os
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image, ImageFilter
import matplotlib
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
        self.gate_contributions = {}
        self.__save_cohort_gate_plots = False
        self.__cohort_gate_layer_idx = -1

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
            self.__save_gate_contributions(slide_ids, attn)
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
        predictions = self.__collect_predictions()
        self.__log_reg_metrics('val', 'reg', self.reg_evaluator.get_metrics, False, predictions)
        self.__log_reg_metrics('val', 'coco', compute_coco_scores, True, predictions)
        self.predictions.clear()

    def on_test_epoch_end(self):
        torch.cuda.empty_cache()
        predictions = self.__collect_predictions()
        self.__log_reg_metrics('test', 'reg', self.reg_evaluator.get_metrics, False, predictions)
        self.__log_reg_metrics('test', 'coco', compute_coco_scores, True, predictions)
        gate_contributions = self.__collect_gate_contributions() if self.__save_cohort_gate_plots else {}
        if self.__is_global_zero():
            self.__write_predictions(predictions)
            if self.__save_cohort_gate_plots:
                self.__write_cohort_gate_contribution_plots(gate_contributions)
        self.predictions.clear()
        self.gate_contributions.clear()

    def enable_cohort_gate_plots(self, layer_idx=-1):
        self.__save_cohort_gate_plots = True
        self.__cohort_gate_layer_idx = layer_idx

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

    def __save_gate_contributions(self, slide_ids, attn_maps):
        if not self.__save_cohort_gate_plots:
            return
        if not isinstance(attn_maps, dict) or attn_maps.get("fusion") is None:
            return

        fusion = attn_maps["fusion"][self.__cohort_gate_layer_idx].detach().cpu().float()
        if fusion.dim() != 5:
            raise ValueError(
                f"Expected fusion weights to have shape [B, T, H, 3, D_head], got {tuple(fusion.shape)}."
            )

        contributions = fusion.mean(dim=(1, 2, 4)).numpy()  # [B, 3]
        for i, slide_id in enumerate(slide_ids):
            self.gate_contributions[slide_id] = {
                "patch": float(contributions[i, 0]),
                "slide": float(contributions[i, 1]),
                "concept": float(contributions[i, 2]),
            }

    def __collect_predictions(self):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return dict(self.predictions)

        gathered_predictions = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_predictions, dict(self.predictions))

        predictions = {}
        for rank_predictions in gathered_predictions:
            if rank_predictions:
                predictions.update(rank_predictions)
        return predictions

    def __collect_gate_contributions(self):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return dict(self.gate_contributions)

        gathered_contributions = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_contributions, dict(self.gate_contributions))

        gate_contributions = {}
        for rank_contributions in gathered_contributions:
            if rank_contributions:
                gate_contributions.update(rank_contributions)
        return gate_contributions

    def __is_global_zero(self):
        trainer = getattr(self, "trainer", None)
        return trainer is None or trainer.is_global_zero

    def __write_predictions(self, predictions):
        os.makedirs(self.__results_dir, exist_ok=True)
        prediction_records = [
            {
                'id': slide_id,
                'predicted': prediction['pred'],
                'ground_trurth': prediction['target']
            }
            for slide_id, prediction in predictions.items()
        ]
        write_json_file(prediction_records, f'{self.__results_dir}/predictions.json')

    def __write_cohort_gate_contribution_plots(self, gate_contributions):
        os.makedirs(self.__results_dir, exist_ok=True)
        records = [
            {"id": slide_id, **contributions}
            for slide_id, contributions in gate_contributions.items()
        ]
        write_json_file(records, f'{self.__results_dir}/gate_contributions.json')

        if not records:
            return

        modalities = ["patch", "slide", "concept"]
        data = [[record[modality] for record in records] for modality in modalities]
        self.__save_gate_distribution_plot(
            data,
            modalities,
            f'{self.__results_dir}/gate_contribution_boxplot.png',
            plot_type="box",
        )
        self.__save_gate_distribution_plot(
            data,
            modalities,
            f'{self.__results_dir}/gate_contribution_violinplot.png',
            plot_type="violin",
        )

    def __save_gate_distribution_plot(self, data, labels, output_path, plot_type):
        fig = Figure(figsize=(6.0, 4.0), dpi=160)
        FigureCanvas(fig)
        ax = fig.add_subplot(111)

        if plot_type == "box":
            ax.boxplot(data, labels=[label.title() for label in labels], patch_artist=True)
            ax.set_title("Gate contribution distribution")
        elif plot_type == "violin":
            parts = ax.violinplot(data, showmeans=True, showmedians=True)
            for body in parts["bodies"]:
                body.set_facecolor("#4c78a8")
                body.set_edgecolor("#1f2933")
                body.set_alpha(0.55)
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels([label.title() for label in labels])
            ax.set_title("Gate contribution density")
        else:
            raise ValueError(f"Unknown gate distribution plot type: {plot_type}")

        ax.set_ylabel("Mean gate contribution")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(output_path)


    def __log_reg_metrics(self, stage, metric_type, evaluate_fn, prog_bar, predictions):
        pred_texts = []
        target_texts = []
        for slide_id in predictions:
            pred_texts.append(predictions[slide_id]['pred'])
            target_texts.append(predictions[slide_id]['target'])

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
                                          output_dir=None, layer_idx=-1, thumbnail_path=None, wsi_size=None):
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

        thumbnail_path = Path(thumbnail_path) if thumbnail_path else self._find_thumbnail_path(slide_id)
        overlay_paths = self._save_attention_overlays(
            slide_id=slide_id,
            thumbnail_path=thumbnail_path,
            attn_maps=attn_maps,
            output_dir=case_dir,
            layer_idx=layer_idx,
            wsi_size=wsi_size,
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

    def _save_attention_overlays(self, slide_id, thumbnail_path, attn_maps, output_dir, layer_idx=-1, wsi_size=None):
        overlays = {}
        if not isinstance(attn_maps, dict):
            return overlays

        coords = self._load_patch_coords(slide_id)
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
            attn, coords = self._align_attention_with_coords(attn, coords, modality)
            self._render_attention_overlay(thumbnail_path, attn, overlay_path, coords=coords, wsi_size=wsi_size)
            overlays[modality] = overlay_path

        return overlays

    def _align_attention_with_coords(self, scores, coords, modality):
        if coords is None:
            raise ValueError("Patch coordinates are required for attention heatmap overlays.")

        coords = np.asarray(coords)
        if len(coords) == scores.numel():
            return scores, coords

        original_token_count = len(coords) + 1  # learned prompt + original patch tokens
        padded_token_count = int(np.ceil(np.sqrt(original_token_count)) ** 2)
        if scores.numel() == padded_token_count:
            # Encoder padding duplicates tokens after [prompt, patch_1, ...].
            # Keep only attention over original patch tokens and drop prompt/padding.
            return scores[1:original_token_count], coords

        raise ValueError(
            f"Cannot overlay {modality} attention: attention token count {scores.numel()} does not match "
            f"patch coordinate count {len(coords)} or expected padded count {padded_token_count}."
        )

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

    def _render_attention_overlay(self, thumbnail_path, scores, output_path, coords=None, wsi_size=None):
        if thumbnail_path is None:
            raise FileNotFoundError(
                "Could not locate a thumbnail for overlay rendering."
            )

        base_image = Image.open(thumbnail_path).convert("RGB")
        heatmap = self._attention_to_heatmap(scores, base_image.size, coords=coords, wsi_size=wsi_size)
        blended = Image.blend(base_image, heatmap, alpha=0.45)
        blended.save(output_path)

    def _attention_to_heatmap(self, scores, size, coords=None, wsi_size=None):
        scores = scores.detach().cpu().float().numpy().reshape(-1)
        if scores.size == 0:
            return Image.new("RGB", size, color=(0, 0, 0))

        if coords is None or len(coords) != scores.size:
            raise ValueError(
                f"Cannot render heatmap: score count {scores.size} does not match coordinate count "
                f"{0 if coords is None else len(coords)}."
            )

        coords = np.asarray(coords)
        if coords.ndim < 2 or coords.shape[1] < 2:
            raise ValueError(f"Expected coordinates with shape [N, 2+], got {coords.shape}.")

        canvas = np.zeros((size[1], size[0]), dtype=np.float32)
        x = coords[:, 0].astype(np.float32)
        y = coords[:, 1].astype(np.float32)
        if wsi_size:
            xi = np.clip((x / max(wsi_size[0], 1) * (size[0] - 1)).round().astype(np.int32), 0, size[0] - 1)
            yi = np.clip((y / max(wsi_size[1], 1) * (size[1] - 1)).round().astype(np.int32), 0, size[1] - 1)
        else:
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)
            xi = np.clip((x * (size[0] - 1)).round().astype(np.int32), 0, size[0] - 1)
            yi = np.clip((y * (size[1] - 1)).round().astype(np.int32), 0, size[1] - 1)
        radius = self._infer_heatmap_radius(xi, yi, size)
        self._splat_scores(canvas, xi, yi, scores, radius)
        canvas = canvas - canvas.min()
        denom = canvas.max()
        if denom > 0:
            canvas = canvas / denom

        colored = self._apply_colormap(canvas)[..., :3]
        colored = (colored * 255).astype(np.uint8)
        heatmap = Image.fromarray(colored)
        return heatmap.filter(ImageFilter.GaussianBlur(radius=6))

    def _apply_colormap(self, values, name="jet"):
        if hasattr(matplotlib, "colormaps"):
            return matplotlib.colormaps[name](values)

        from matplotlib import pyplot as plt
        return plt.get_cmap(name)(values)

    def _infer_heatmap_radius(self, xi, yi, size):
        if xi.size < 2:
            return max(2, min(size) // 200)

        unique_x = np.unique(xi)
        unique_y = np.unique(yi)
        dx = np.diff(np.sort(unique_x))
        dy = np.diff(np.sort(unique_y))
        spacing = np.concatenate([dx[dx > 0], dy[dy > 0]])
        if spacing.size == 0:
            return max(2, min(size) // 200)

        return max(2, int(round(np.median(spacing) / 2)))

    def _splat_scores(self, canvas, xi, yi, scores, radius):
        height, width = canvas.shape
        for x, y, score in zip(xi, yi, scores):
            x0 = max(0, x - radius)
            x1 = min(width, x + radius + 1)
            y0 = max(0, y - radius)
            y1 = min(height, y + radius + 1)
            canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], score)

    def _load_patch_coords(self, slide_id):
        args = self.model.encoder_decoder.args
        data_dir = getattr(args, "embeddings_path_2", None)
        if not data_dir:
            raise ValueError("args.embeddings_path_2 is required to load patch coordinates.")

        h5_path = Path(data_dir) / f"{slide_id}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Could not find patch embedding H5 for {slide_id}: {h5_path}")

        try:
            import h5py
        except Exception as exc:
            raise ImportError("h5py is required to load patch coordinates.") from exc

        coord_keys = ("coords", "patch_coords", "bag_coords")
        with h5py.File(h5_path, "r") as h5_file:
            for key in coord_keys:
                if key in h5_file:
                    coords = h5_file[key][:]
                    return coords
        raise KeyError(f"No coordinate dataset found in {h5_path}. Tried keys: {coord_keys}")

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
