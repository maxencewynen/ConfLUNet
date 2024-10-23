import os
import time
import wandb
import warnings
from os.path import join as pjoin
from typing import Callable, Dict, Union

from monai.data import DataLoader
from nnunetv2.paths import  nnUNet_results
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

from conflunet.architecture.nnconflunet import *
from conflunet.architecture.utils import get_model
from conflunet.evaluation.semantic_segmentation import dice_metric, dice_norm_metric
from conflunet.evaluation.instance_segmentation import panoptic_quality
from conflunet.inference.predictors.base_predictor import Predictor
from conflunet.postprocessing.small_instances_removal import remove_small_lesions_from_instance_segmentation as remove_small_lesions
from conflunet.utilities.planning_and_configuration import load_dataset_and_configuration
from conflunet.dataloading.dataloaders import (
    get_train_dataloader_from_dataset_id_and_fold,
    get_val_dataloader_from_dataset_id_and_fold,
    get_full_val_dataloader_from_dataset_id_and_fold)
from conflunet.training.utils import get_default_device, seed_everything

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


class TrainingPipeline:
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
                 weight_decay=3e-5,
                 momentum=0.99,
                 semantic: bool = False,
                 predictors: List[Predictor] = None,
                 metrics_to_track: Dict[str, Tuple[Callable, bool]] = None
                 ):
        self.dataset_id = dataset_id
        self.fold = fold
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.dice_loss_weight = dice_loss_weight
        self.focal_loss_weight = focal_loss_weight
        self.seed_val = seed
        self.val_interval = val_interval
        self.actual_val_interval = actual_val_interval
        self.wandb_ignore = wandb_ignore
        self.wandb_project = wandb_project
        self.model_name = model_name
        self.force_restart = force_restart
        self.debug = debug
        self.save_predictions = save_predictions
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.semantic = semantic
        self.predictors = predictors
        self.metrics_to_track = metrics_to_track
        self.save_dir = self.full_validation_save_dir = self.patches_save_dir = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.start_epoch = None
        self.best_metrics = None
        self.loss_fn = None
        self.wandb_run_id = None

        seed_everything(self.seed_val)
        self.device = get_default_device()
        torch.multiprocessing.set_sharing_strategy('file_system')

        # Dataset and configuration setup
        self.dataset_name, self.plans_manager, self.configuration, self.n_channels = \
            load_dataset_and_configuration(self.dataset_id)

        # Dataloaders
        self.train_loader, self.val_loader = self.get_dataloaders()
        self.full_val_loader = get_full_val_dataloader_from_dataset_id_and_fold(self.dataset_id, self.fold, self.num_workers)

        self._edit_name()

        # Directory setup
        self.setup_save_dir()

        # Checkpoint filename
        self.checkpoint_filename = os.path.join(self.save_dir, f"checkpoint_final.pth")

        # Model and optimizer
        self.set_model_optimizer_and_scheduler()

        # Loss functions
        self.loss_fn = self.get_loss_functions()

        # Metrics and tracking
        self.best_val_loss = np.inf

        if self.metrics_to_track is None:
            self.metrics_to_track = {
                "Validation Metrics/Dice Score": (dice_metric, True), # True means it needs semantic segmentation
                "Validation Metrics/Normalized Dice": (dice_norm_metric, True),
                "Validation Metrics/Panoptic Quality": (panoptic_quality, False), # False means it needs instance segmentation
            }

        self.scaler = torch.cuda.amp.GradScaler()

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = get_train_dataloader_from_dataset_id_and_fold(self.dataset_id, self.fold,
                                                                     num_workers=self.num_workers,
                                                                     cache_rate=self.cache_rate,
                                                                     seed_val=self.seed_val)

        if self.debug:
            val_loader = [next(iter(train_loader))]
            train_loader = val_loader * len(train_loader)
        else:
            val_loader = get_val_dataloader_from_dataset_id_and_fold(self.dataset_id, self.fold,
                                                                     num_workers=self.num_workers,
                                                                     cache_rate=self.cache_rate,
                                                                     seed_val=self.seed_val)
        return train_loader, val_loader

    def _edit_name(self) -> None:
        if self.debug:
            self.model_name = f"DEBUG_{self.model_name}"

    def setup_save_dir(self) -> None:
        self.save_dir = pjoin(nnUNet_results, self.dataset_name, self.model_name, 'fold_%d' % self.fold)
        os.makedirs(self.save_dir, exist_ok=True)

        if self.save_predictions or self.debug:
            self.patches_save_dir = pjoin(self.save_dir, 'saved_patches')
            os.makedirs(self.patches_save_dir, exist_ok=True)
            self.full_validation_save_dir = pjoin(self.save_dir, 'saved_predictions')
            os.makedirs(self.full_validation_save_dir, exist_ok=True)

    def set_model_optimizer_and_scheduler(self) -> Tuple[nn.Module, torch.optim.Optimizer, object, int, str]:
        self.wandb_run_id = None
        self.model = get_model(self.configuration, self.n_channels, self.semantic).to(self.device)

        # Initialize optimizer and scheduler according to nnunet
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate,
                                         weight_decay=self.weight_decay, momentum=self.momentum)
        self.lr_scheduler = PolyLRScheduler(self.optimizer, self.learning_rate, self.n_epochs)
        self.start_epoch = 0

        if os.path.exists(self.checkpoint_filename) and not self.force_restart:
            checkpoint = torch.load(self.checkpoint_filename)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
            self.best_metrics = checkpoint["best_metrics"]

            if not self.wandb_ignore:
                self.wandb_run_id = checkpoint['wandb_run_id']
                wandb.init(project=self.wandb_project, mode="online", name=self.model_name, resume="must", id=self.wandb_run_id)
            print(f"\nResuming training: (epoch {checkpoint['epoch']})\nLoaded checkpoint '{self.checkpoint_filename}'\n")

        elif not self.wandb_ignore:
            wandb.login()
            wandb.init(project=self.wandb_project, mode="online", name=self.model_name)
            self.wandb_run_id = wandb.run.id

    def get_loss_functions(self) -> Callable:
        raise NotImplementedError("Subclass must implement this method")

    def compute_loss(self, model_outputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        raise NotImplementedError("Subclass must implement this method")

    def save_train_patch_debug(self, batch_inputs: Tuple[torch.Tensor],
                               model_outputs: Union[torch.Tensor, Tuple[torch.Tensor]], epoch: int) -> None:
        raise NotImplementedError("Subclass must implement this method")

    def save_val_patch_debug(self, batch_inputs: Tuple[torch.Tensor],
                             model_outputs: Union[torch.Tensor, Tuple[torch.Tensor]], epoch: int) -> None:
        raise NotImplementedError("Subclass must implement this method")

    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        epoch_logs = self.initialize_epoch_logs()
        start_epoch_time = time.time()
        for batch_idx, batch_data in enumerate(self.train_loader):
            start_batch_time = time.time()
            # Prepare the batch (specific to subclass)
            img, outputs = self.prepare_batch(batch_data)

            # Forward pass
            model_outputs = self.model(img)
            loss_values = self.compute_loss(model_outputs, outputs)
            total_loss = loss_values[0]

            # Update the loss values for logging
            self.update_loss_values(epoch_logs, loss_values)

            # Perform optimizer step
            self.optimizer_step(total_loss)

            if self.debug and epoch == 0 and batch_idx == 0:
                self.save_train_patch_debug((img, outputs), model_outputs, epoch)

            self.print_batch_progress(batch_idx, total_loss, start_batch_time)

        self.average_epoch_logs(epoch_logs)

        self.print_epoch_summary(epoch_logs, start_epoch_time, epoch)

        if not self.wandb_ignore:
            wandb.log({**epoch_logs, **{'Learning rate': self.optimizer.param_groups[0]['lr']}}, step=epoch)

    def initialize_epoch_logs(self) -> dict:
        raise NotImplementedError

    def initialize_val_logs(self) -> dict:
        raise NotImplementedError

    def average_epoch_logs(self, epoch_logs: dict) -> None:
        for key in epoch_logs:
            epoch_logs[key] /= len(self.train_loader)

    def average_val_logs(self, val_logs: dict) -> None:
        for key in val_logs:
            val_logs[key] /= len(self.val_loader)

    def average_full_val_metrics(self, val_metrics: dict) -> None:
        for key in val_metrics:
            val_metrics[key] /= len(self.full_val_loader)

    def prepare_batch(self, batch_data: dict) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor]]]:
        raise NotImplementedError("Subclass must implement this method")

    @staticmethod
    def update_loss_values(epoch_loss_values: dict, loss_values: Tuple[torch.Tensor]) -> None:
        assert len(epoch_loss_values) == len(loss_values), "Loss values must have the same length as the epoch loss values"
        for key, loss in zip(epoch_loss_values.keys(), loss_values):
            epoch_loss_values[key] += loss.item()

    def optimizer_step(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def print_batch_progress(self, batch_idx: int, loss: torch.Tensor, start_batch_time: float) -> None:
        elapsed_time = time.time() - start_batch_time
        print(f"Batch {batch_idx + 1}/{len(self.train_loader)}, train_loss: {loss.item():.4f} "
              f"(elapsed time: {int(elapsed_time // 60)}min {int(elapsed_time % 60)}s)")

    @staticmethod
    def print_epoch_summary(epoch_loss_values: dict, start_epoch_time: float, epoch: int) -> None:
        elapsed_epoch_time = time.time() - start_epoch_time
        print(f"Epoch {epoch + 1} took {int(elapsed_epoch_time // 60)}min {int(elapsed_epoch_time % 60)}s")
        print(f"Epoch average loss: {epoch_loss_values['Training Loss/Total Loss']:.4f}")

    def validate_epoch(self, epoch: int) -> None:
        self.model.eval()
        avg_val_losses = self.initialize_val_logs()
        start_validation_time = time.time()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                # Prepare the batch (specific to subclass)
                img, outputs = self.prepare_batch(batch_data)

                # Forward pass
                model_outputs = self.model(img)
                loss_values = self.compute_loss(model_outputs, outputs)

                # Update the loss values for logging
                self.update_loss_values(avg_val_losses, loss_values)

                if self.debug and batch_idx == 0:
                    self.save_val_patch_debug((img, outputs), model_outputs, epoch)

            self.average_val_logs(avg_val_losses)

            if not self.wandb_ignore:
                wandb.log(avg_val_losses, step=epoch)

            self.print_val_summary(avg_val_losses, start_validation_time)

    def full_validation(self, epoch: int) -> None:
        if self.metrics_to_track is None:
            return

        self.model.eval()
        start_full_validation_time = time.time()
        predictors = [self.predictors] if isinstance(self.predictors, Predictor) else self.predictors

        for predictor in predictors:
            avg_val_metrics = {k: 0 for k in self.best_metrics[predictor.postprocessor.name].keys()}
            start_this_predictor = time.time()
            predictions_loader = predictor.get_predictions_loader(self.full_val_loader, self.model)
            for gt, pred in zip(self.full_val_loader, predictions_loader):
                self.update_metrics(avg_val_metrics, gt, pred)
                if self.save_predictions:
                    predictor.save_predictions(pred)

            print(f"Full dataset prediction by Predictor {predictor.__class__.__name__} took {time.time() - start_this_predictor:.2f} seconds")

            self.average_full_val_metrics(avg_val_metrics)
            for metric in avg_val_metrics:
                if avg_val_metrics[metric] > self.best_metrics[predictor.postprocessor.name][metric]:
                    self.best_metrics[predictor.postprocessor.name][metric] = avg_val_metrics[metric]
                    m = metric.replace("Validation Metrics/", "").replace(" ", "_")
                    checkpoint_filename = os.path.join(self.save_dir, f"checkpoint_best_{m}_{predictor.postprocessor.name}.pth")
                    self.save_checkpoint(epoch, checkpoint_filename=checkpoint_filename)

            if not self.wandb_ignore:
                wandb.log({k + '_' + predictor.postprocessor.name: v for k, v in avg_val_metrics.items()}, step=epoch)

        print(f"Full dataset prediction by all Predictors took {time.time() - start_full_validation_time:.2f} seconds")

    def update_metrics(self, avg_val_metrics: dict, gt: dict, pred: dict) -> None:
        assert pred['name'] == gt['name'][0], f"GT and Prediction don't match {pred['name']} != {gt['name'][0]}"
        print(f"[INFO] Computing metrics ... ({pred['name']})")
        start_metric_computation_time = time.time()
        instance_seg_pred = np.squeeze(pred['instance_seg_pred'].detach().cpu().numpy())
        semantic_pred_binary = np.squeeze(pred['semantic_pred_binary'].detach().cpu().numpy())
        gt_instance_seg = remove_small_lesions(np.squeeze(gt['instance_seg'].detach().cpu().numpy()),
                                               self.configuration.spacing)
        gt_semantic = (gt_instance_seg > 0).astype(np.int16)
        
        for metric_name, (metric_fn, semantic) in self.metrics_to_track.items():
            if semantic:
                avg_val_metrics[metric_name] += metric_fn(semantic_pred_binary, gt_semantic)
            else:
                avg_val_metrics[metric_name] += metric_fn(instance_seg_pred, gt_instance_seg)

        print(f"[INFO] Metric computation for {pred['name']} took {time.time() - start_metric_computation_time:.2f} seconds")

    @staticmethod
    def print_val_summary(avg_val_losses: dict, start_validation_time: float, full: bool = False) -> None:
        val_elapsed_time = time.time() - start_validation_time
        if full:
            print("Full", end=" ")
        print(f"Validation took {int(val_elapsed_time // 60)}min {int(val_elapsed_time % 60)}s")
        for key in avg_val_losses:
            print(f"{key}: {avg_val_losses[key]:.4f}")

    def save_checkpoint(self, epoch: int, checkpoint_filename: str = None) -> None:
        if not self.debug and not self.wandb_ignore:
            checkpoint_filename = checkpoint_filename or self.checkpoint_filename
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'wandb_run_id': self.wandb_run_id,
                'scheduler': self.lr_scheduler.state_dict(),
                'best_metrics': self.best_metrics
            }, checkpoint_filename)

    def run_training(self) -> None:
        for epoch in range(self.start_epoch, self.n_epochs):
            print("-" * 10, f"\nEpoch {epoch + 1}/{self.n_epochs}")
            self.train_epoch(epoch)

            if (epoch + 1) % self.val_interval == 0:
                self.validate_epoch(epoch)

            self.lr_scheduler.step()

            if (epoch + 1) % self.actual_val_interval == 0:
                self.full_validation(epoch)

            self.save_checkpoint(epoch)
