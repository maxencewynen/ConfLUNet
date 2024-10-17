import os
import time
import wandb
import warnings
from os.path import join as pjoin

from nnunetv2.paths import  nnUNet_results
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

from conflunet.architecture.nnconflunet import *
from conflunet.architecture.utils import get_model
from conflunet.utilities.planning_and_configuration import load_dataset_and_configuration
from conflunet.dataloading.dataloaders import get_train_dataloader_from_dataset_id_and_fold
from conflunet.dataloading.dataloaders import get_val_dataloader_from_dataset_id_and_fold
from conflunet.training.utils import get_default_device, seed_everything, save_patch

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
        self.save_dir = None
        self.patches_save_dir = None

        self.device = get_default_device()
        torch.multiprocessing.set_sharing_strategy('file_system')

        # Dataset and configuration setup
        self.dataset_name, self.plans_manager, self.configuration, self.n_channels = \
            load_dataset_and_configuration(self.dataset_id)

        # Dataloaders
        self.train_loader, self.val_loader = self.get_dataloaders()

        self._edit_name()

        # Directory setup
        self.setup_save_dir()

        # Checkpoint filename
        self.checkpoint_filename = os.path.join(self.save_dir, f"checkpoint_final.pth")

        # Model and optimizer
        self.model, self.optimizer, self.lr_scheduler, self.start_epoch, self.wandb_run_id = \
            self.get_model_optimizer_and_scheduler()

        # Loss functions
        self.loss_fn = self.get_loss_functions()

        # Metrics and tracking
        self.best_val_loss = np.inf
        self.best_metrics = {metric: -np.inf for metric in
                             ["Validation Metrics/Dice Score", "Validation Metrics/Normalized Dice"]}

        self.scaler = torch.cuda.amp.GradScaler()

    def get_dataloaders(self):
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

    def _edit_name(self):
        if self.debug:
            self.model_name = f"DEBUG_{self.model_name}"

    def setup_save_dir(self):
        self.save_dir = pjoin(nnUNet_results, self.dataset_name, self.model_name, 'fold_%d' % self.fold)
        os.makedirs(self.save_dir, exist_ok=True)

        if self.save_predictions or self.debug:
            self.patches_save_dir = pjoin(self.save_dir, 'saved_patches')
            os.makedirs(self.patches_save_dir, exist_ok=True)

    def get_model_optimizer_and_scheduler(self,) -> Tuple[nn.Module, torch.optim.Optimizer, object, int, str]:
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

            if not self.wandb_ignore:
                self.wandb_run_id = checkpoint['wandb_run_id']
                wandb.init(project=self.wandb_project, mode="online", name=self.model_name, resume="must", id=self.wandb_run_id)
            print(f"\nResuming training: (epoch {checkpoint['epoch']})\nLoaded checkpoint '{self.checkpoint_filename}'\n")

        elif not self.wandb_ignore:
            wandb.login()
            wandb.init(project=self.wandb_project, mode="online", name=self.model_name)
            self.wandb_run_id = wandb.run.id

    def get_loss_functions(self):
        raise NotImplementedError("Subclass must implement this method")

    def compute_loss(self, model_outputs, batch_inputs):
        raise NotImplementedError("Subclass must implement this method")

    def save_train_patch_debug(self, batch_inputs, model_outputs, epoch):
        raise NotImplementedError("Subclass must implement this method")

    def save_val_patch_debug(self, batch_inputs, model_outputs, epoch):
        raise NotImplementedError("Subclass must implement this method")

    def train_epoch(self, epoch):
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
                self.save_train_patch_debug(batch_data, model_outputs, epoch)

            self.print_batch_progress(batch_idx, total_loss, start_batch_time)

        self.average_epoch_logs(epoch_logs)

        self.print_epoch_summary(epoch_logs, len(self.train_loader), start_epoch_time)

        if not self.wandb_ignore:
            wandb.log({**epoch_logs, **{'Learning rate': self.optimizer.param_groups[0]['lr']}}, step=epoch)

    def initialize_epoch_logs(self):
        raise NotImplementedError

    def initialize_val_metrics(self):
        raise NotImplementedError

    def average_epoch_logs(self, epoch_logs):
        for key in epoch_logs:
            epoch_logs[key] /= len(self.train_loader)

    def average_val_logs(self, val_logs):
        for key in val_logs:
            val_logs[key] /= len(self.val_loader)

    def prepare_batch(self, batch_data):
        raise NotImplementedError("Subclass must implement this method")

    @staticmethod
    def update_loss_values(epoch_loss_values, loss_values):
        for key, loss in zip(epoch_loss_values.keys(), loss_values):
            epoch_loss_values[key] += loss.item()

    def optimizer_step(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def print_batch_progress(self, batch_idx, loss, start_batch_time):
        elapsed_time = time.time() - start_batch_time
        print(f"Batch {batch_idx + 1}/{len(self.train_loader)}, train_loss: {loss.item():.4f} "
              f"(elapsed time: {int(elapsed_time // 60)}min {int(elapsed_time % 60)}s)")

    @staticmethod
    def print_epoch_summary(epoch_loss_values, n_batches, start_epoch_time):
        for key in epoch_loss_values:
            epoch_loss_values[key] /= n_batches
        print(f"Epoch average loss: {epoch_loss_values['Training Loss/Total Loss']:.4f}")

    def validate_epoch(self, epoch):
        self.model.eval()
        avg_val_metrics = self.initialize_val_metrics()
        start_validation_time = time.time()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                # Prepare the batch (specific to subclass)
                img, outputs = self.prepare_batch(batch_data)

                # Forward pass
                model_outputs = self.model(img)
                loss_values = self.compute_loss(model_outputs, outputs)

                # Update the loss values for logging
                self.update_loss_values(avg_val_metrics, loss_values)

                if self.debug and batch_idx == 0:
                    self.save_val_patch_debug(batch_data, model_outputs, epoch)

            self.average_val_logs(avg_val_metrics)

            val_elapsed_time = time.time() - start_validation_time
            print(f"Validation took {int(val_elapsed_time // 60)}min {int(val_elapsed_time % 60)}s\n"
                  f"Validation Loss/Total Loss: {avg_val_metrics['Validation Loss/Total Loss']:.4f}")
            for key in avg_val_metrics:
                print(f"{key}: {avg_val_metrics[key]:.4f}")

            if not self.wandb_ignore:
                wandb.log(avg_val_metrics, step=epoch)

    @staticmethod
    def print_val_summary(avg_val_metrics, epoch):
        print(f"Validation Loss/Total Loss: {avg_val_metrics['Validation Loss/Total Loss']:.4f}")
        for key in avg_val_metrics:
            print(f"{key}: {avg_val_metrics[key]:.4f}")

    def save_checkpoint(self, epoch):
        if not self.debug and not self.wandb_ignore:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'wandb_run_id': self.wandb_run_id,
                'scheduler': self.lr_scheduler.state_dict()
            }, self.checkpoint_filename)

    def run_training(self):
        for epoch in range(self.start_epoch, self.n_epochs):
            print("-" * 10, f"\nEpoch {epoch + 1}/{self.n_epochs}")
            self.train_epoch(epoch)

            if (epoch + 1) % self.val_interval == 0:
                self.validate_epoch(epoch)

            self.lr_scheduler.step()

            # if (epoch + 1) % self.actual_val_interval == 0:
            #     self.full_validation(epoch)

            self.save_checkpoint(epoch)
