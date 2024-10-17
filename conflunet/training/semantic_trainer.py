import numpy as np
import torch

from conflunet.training.losses import SemanticSegmentationLoss
from conflunet.training.trainer import TrainingPipeline
from conflunet.training.utils import save_patch


class SemanticTrainer(TrainingPipeline):
    def __init__(self,
                 dataset_id: int = 321,
                 fold: int = 0,
                 num_workers: int = 12,
                 cache_rate: float = 0.0,
                 learning_rate: float = 1e-2,
                 n_epochs: int = 1500,
                 dice_loss_weight: float = 0.5,
                 focal_loss_weight: float = 1.0,
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

    def _edit_name(self):
        self.model_name = f"SEMANTIC_{self.model_name}"
        if self.debug:
            self.model_name = f"DEBUG_{self.model_name}"

    def prepare_batch(self, batch_data):
        inputs = batch_data["img"].to(self.device)
        labels = batch_data["seg"].type(torch.LongTensor).to(self.device)
        return inputs, labels

    def get_loss_functions(self):
        return SemanticSegmentationLoss(dice_loss_weight=self.dice_loss_weight,
                                        focal_loss_weight=self.focal_loss_weight)

    def compute_loss(self, model_outputs, output):
        return self.loss_fn(output, model_outputs)

    def save_train_patch_debug(self, batch_inputs, model_outputs, epoch):
        save_patch(batch_inputs[0].cpu().numpy(), f'Epoch-{epoch}_train_image', self.patches_save_dir)
        save_patch(batch_inputs[1].cpu().numpy().astype(np.int16), f'Epoch-{epoch}_train_labels', self.patches_save_dir)

    def save_val_patch_debug(self, batch_inputs, model_outputs, epoch):
        save_patch(batch_inputs[0].cpu().numpy(), f'Epoch-{epoch}_val_image', self.patches_save_dir)
        save_patch(batch_inputs[1].cpu().numpy(), f'Epoch-{epoch}_val_labels', self.patches_save_dir)
        save_patch(model_outputs[0], f"Epoch-{epoch}_pred_segmentation_proba", self.patches_save_dir)

    def initialize_epoch_logs(self):
        return {
            'Training Loss/Total Loss': 0,
            'Training Segmentation Loss/Dice Loss': 0,
            'Training Segmentation Loss/Focal Loss': 0,
        }

    def initialize_val_metrics(self):
        return {
            'Validation Loss/Total Loss': 0,
            'Validation Segmentation Loss/Dice Loss': 0,
            'Validation Segmentation Loss/Focal Loss': 0,
        }