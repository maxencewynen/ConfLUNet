import torch
import numpy as np
from typing import Tuple, Callable
from torch.nn import SmoothL1Loss, L1Loss

from conflunet.postprocessing.instance import ConfLUNetPostprocessor
from conflunet.training.losses import ConfLUNetLoss, SemanticSegmentationLoss
from conflunet.training.trainer import TrainingPipeline
from conflunet.training.utils import save_patch
from conflunet.inference.predictors.instance import ConfLUNetPredictor


class ConfLUNetTrainer(TrainingPipeline):
    def __init__(self,
                 dataset_id: int = 321,
                 fold: int = 0,
                 num_workers: int = 12,
                 cache_rate: float = 0.0,
                 learning_rate: float = 1e-2,
                 n_epochs: int = 1500,
                 dice_loss_weight: float = 0.5,
                 focal_loss_weight: float = 1.0,
                 seg_loss_weight: float = 1.0,
                 heatmap_loss_weight: float = 1.0,
                 offsets_loss_weight: float = 1.0,
                 offsets_loss: str = "l1",
                 seed: int = 1,
                 val_interval: int = 5,
                 actual_val_interval: int = 50,
                 wandb_ignore: bool = False,
                 wandb_project: str = None,
                 model_name: str = None,
                 force_restart: bool = False,
                 debug: bool = False,
                 save_predictions: bool = False,
                 ):
        super().__init__(dataset_id=dataset_id,
                         fold=fold,
                         num_workers=num_workers,
                         cache_rate=cache_rate,
                         learning_rate=learning_rate,
                         n_epochs=n_epochs,
                         dice_loss_weight=dice_loss_weight,
                         focal_loss_weight=focal_loss_weight,
                         seed=seed,
                         val_interval=val_interval,
                         actual_val_interval=actual_val_interval,
                         wandb_ignore=wandb_ignore,
                         wandb_project=wandb_project,
                         model_name=model_name,
                         force_restart=force_restart,
                         debug=debug,
                         save_predictions=save_predictions,
                         semantic=True)

        self.seg_loss_weight = seg_loss_weight
        self.heatmap_loss_weight = heatmap_loss_weight
        self.offsets_loss_weight = offsets_loss_weight
        if offsets_loss == "sl1":
            self.offsets_loss = SmoothL1Loss(reduction='none')
        elif offsets_loss == "l1":
            self.offsets_loss = L1Loss(reduction='none')
        else:
            raise ValueError(f"Offsets loss {self.offsets_loss} not recognized. "
                             f"Choose between 'sl1' and 'l1'.")

        self.predictors = ConfLUNetPredictor(
            plans_manager=self.plans_manager,
            model=self.model,
            postprocessor=ConfLUNetPostprocessor(
                minimum_instance_size=0,
                minimum_size_along_axis=0,
                semantic_threshold=0.5,
                heatmap_threshold=0.1,
                nms_kernel_size=3,
                top_k=100,
                compute_voting=False,
                calibrate_offsets=False,
                device=self.device,
            ),
            output_dir=self.full_validation_save_dir,
            num_workers=self.num_workers,
            save_only_instance_segmentation=False
        ),

        if self.best_metrics is None:
            self.best_metrics = {'ConfLUNet': {metric: -np.inf for metric in self.metrics_to_track.keys()}}

    def _edit_name(self) -> None:
        if self.debug:
            self.model_name = f"DEBUG_{self.model_name}"

    def prepare_batch(self, batch_data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = batch_data["img"].to(self.device)
        labels = batch_data["seg"].type(torch.LongTensor).to(self.device)
        center_heatmap = batch_data["center_heatmap"].to(self.device)
        offsets = batch_data["offsets"].to(self.device)
        return inputs, (labels, center_heatmap, offsets)

    def get_loss_functions(self) -> Callable:
        return ConfLUNetLoss(
            segmentation_loss_weight=self.seg_loss_weight,
            offsets_loss_weight=self.offsets_loss_weight,
            center_heatmap_loss_weight=self.heatmap_loss_weight,
            loss_function_segmentation=SemanticSegmentationLoss(
                dice_loss_weight=self.dice_loss_weight,
                focal_loss_weight=self.focal_loss_weight
            ),
            loss_function_offsets=self.offsets_loss,
        )

    def compute_loss(self, model_outputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        seg_pred, center_heatmap_pred, offsets_pred = model_outputs
        semantic_ref, center_heatmap_ref, offsets_ref = outputs

        return self.loss_fn(
            semantic_pred=seg_pred,
            center_heatmap_pred=center_heatmap_pred,
            offsets_pred=offsets_pred,
            semantic_ref=semantic_ref,
            center_heatmap_ref=center_heatmap_ref,
            offsets_ref=offsets_ref
        )

    def save_train_patch_debug(self, batch_inputs: Tuple[torch.Tensor, torch.Tensor],
                               model_outputs: torch.Tensor | Tuple[torch.Tensor], epoch: int) -> None:
        img, (labels, center_heatmap, offsets) = batch_inputs
        semantic_pred, center_heatmap_pred, offsets_pred = model_outputs
        semantic_pred = semantic_pred[0, 1, :, :, :]

        save_patch(img.detach().cpu().numpy(), f'Epoch-{epoch}_train_image', self.patches_save_dir)
        save_patch(labels.detach().cpu().numpy().astype(np.int16), f'Epoch-{epoch}_train_labels', self.patches_save_dir)
        save_patch(center_heatmap.detach().cpu().numpy(), f'Epoch-{epoch}_train_center_heatmap', self.patches_save_dir)
        save_patch(offsets.detach().cpu().numpy(), f'Epoch-{epoch}_train_offsets', self.patches_save_dir)

        save_patch(semantic_pred.detach().cpu().numpy().astype(np.int16), f'Epoch-{epoch}_train_pred_segmentation_proba', self.patches_save_dir)
        save_patch(center_heatmap_pred.detach().cpu().numpy(), f'Epoch-{epoch}_train_pred_center_heatmap', self.patches_save_dir)
        save_patch(offsets_pred.detach().cpu().numpy(), f'Epoch-{epoch}_train_pred_offsets', self.patches_save_dir)

    def save_val_patch_debug(self, batch_inputs: Tuple[torch.Tensor], model_outputs: torch.Tensor | Tuple[torch.Tensor],
                             epoch: int) -> None:
        img, (labels, center_heatmap, offsets) = batch_inputs
        semantic_pred, center_heatmap_pred, offsets_pred = model_outputs
        semantic_pred = semantic_pred[0, 1, :, :, :]

        save_patch(img.cpu().numpy(), f'Epoch-{epoch}_val_image', self.patches_save_dir)
        save_patch(labels.cpu().numpy().astype(np.int16), f'Epoch-{epoch}_val_labels', self.patches_save_dir)
        save_patch(center_heatmap.cpu().numpy(), f'Epoch-{epoch}_val_center_heatmap', self.patches_save_dir)
        save_patch(offsets.cpu().numpy(), f'Epoch-{epoch}_val_offsets', self.patches_save_dir)

        save_patch(semantic_pred.cpu().numpy(), f"Epoch-{epoch}_val_pred_segmentation_proba", self.patches_save_dir)
        save_patch(center_heatmap_pred.cpu().numpy(), f'Epoch-{epoch}_val_pred_center_heatmap', self.patches_save_dir)
        save_patch(offsets_pred.cpu().numpy(), f'Epoch-{epoch}_val_pred_offsets', self.patches_save_dir)

    def initialize_epoch_logs(self) -> dict:
        return {
            'Training Loss/Total Loss': 0,
            'Training Segmentation Loss/Dice Loss': 0,
            'Training Segmentation Loss/Focal Loss': 0,
            'Training Loss/Segmentation Loss': 0,
            'Training Loss/Center Prediction Loss': 0,
            'Training Loss/Offsets Loss': 0,
        }

    def initialize_val_logs(self) -> dict:
        return {
            'Validation Loss/Total Loss': 0,
            'Validation Segmentation Loss/Dice Loss': 0,
            'Validation Segmentation Loss/Focal Loss': 0,
            'Validation Loss/Segmentation Loss': 0,
            'Validation Loss/Center Prediction Loss': 0,
            'Validation Loss/Offsets Loss': 0,
        }
