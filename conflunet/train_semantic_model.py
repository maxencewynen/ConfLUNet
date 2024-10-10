import argparse
import os
import torch
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from torch import nn
from monai.losses import DiceLoss
from monai.utils import set_determinism
import numpy as np
import random

from conflunet.preprocessing.preprocess import PlansManagerInstanceSeg
from conflunet.architecture.conflunet import *
from conflunet.architecture.nnconflunet import *
from conflunet.dataloading.dataloaders import (
    get_train_dataloader_from_dataset_id_and_fold,
    get_val_dataloader_from_dataset_id_and_fold
)
import wandb
from os.path import join as pjoin
from metrics import *
import time
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser(description='Get all command line arguments.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('--dataset_id', type=int, default=321, help='Specify the dataset id')
parser.add_argument('--fold', type=int, default=0, help='Specify the fold')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')
parser.add_argument('--cache_rate', default=0, type=float)

# trainining
parser.add_argument('--learning_rate', type=float, default=1e-2, help='Specify the initial learning rate')
parser.add_argument('--n_epochs', type=int, default=1500, help='Specify the number of epochs to train for')

# initialisation
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')

# logging
parser.add_argument('--val_interval', type=int, default=5, help='Validation every n-th epochs')

parser.add_argument('--wandb_ignore', action="store_true", default=False, help='Whether to ignore this run in wandb')
parser.add_argument('--wandb_project', type=str, default='ConfLUNet_Semantic', help='wandb project name')
parser.add_argument('--name', default="idiot without a name", help='Wandb run name')
parser.add_argument('--force_restart', default=False, action='store_true',
                    help="force the training to restart at 0 even if a checkpoint was found")

# debugging
parser.add_argument('--debug', default=False, action='store_true',
                    help="Debug mode: use a single batch for training and validation")
parser.add_argument('--save_predictions', default=False, action='store_true',
                    help="Debug mode: save predictions of the first batch for each validation step")


def load_checkpoint(model, optimizer, filename):
    print(f"=> Loading checkpoint '{filename}'")
    checkpoint = torch.load(filename)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    wandb_run_id = checkpoint['wandb_run_id']
    print(f"\nResuming training: (epoch {checkpoint['epoch']})\nLoaded checkpoint '{filename}'\n")

    return model, optimizer, start_epoch, wandb_run_id


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        return torch.device('cuda')
    raise Exception("No GPU device was found: I cannot train without GPU.")


def seed_everything(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    set_determinism(seed=seed_val)


def compute_loss(semantic_pred, labels):
    # Initialize losses
    loss_function_dice = DiceLoss(to_onehot_y=True,
                                  softmax=True, sigmoid=False,
                                  include_background=False)
    loss_function_mse = nn.MSELoss()
    if args.offsets_loss == 'l1':
        offset_loss_fn = nn.L1Loss(reduction='none')
    elif args.offsets_loss == 'sl1':
        offset_loss_fn = nn.SmoothL1Loss(reduction='none')
    else:
        raise ValueError(f"Invalid loss function for offsets: {args.offsets_loss}")

    # Initialize other variables and metrics
    gamma_focal = 2.0
    dice_weight = 0.5#1
    focal_weight = 1.0#1
    seg_loss_weight = args.seg_loss_weight
    heatmap_loss_weight = args.heatmap_loss_weight
    offsets_loss_weight = args.offsets_loss_weight

    ### SEGMENTATION LOSS ###
    # Dice loss
    dice_loss = loss_function_dice(semantic_pred, labels)
    # Focal loss
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    ce = ce_loss(semantic_pred, torch.squeeze(labels, dim=1))
    pt = torch.exp(-ce)
    loss2 = (1 - pt) ** gamma_focal * ce
    focal_loss = torch.mean(loss2)
    segmentation_loss = dice_weight * dice_loss + focal_weight * focal_loss

    return segmentation_loss, dice_loss, focal_loss


def save_patch(data, name, save_dir):
    import nibabel as nib
    import numpy as np
    tobesaved = data if len(data.shape) == 3 else np.squeeze(data[0,0,:,:,:])
    nib.save(nib.Nifti1Image(tobesaved, np.eye(4)),
            pjoin(save_dir, f'{name}.nii.gz'))


def get_model_optimizer_and_scheduler(args, configuration, n_channels, device, checkpoint_filename):
    wandb_run_id = None
    # Initialize model
    if os.path.exists(checkpoint_filename) and not args.force_restart:
        model = get_network_from_plans(
            configuration.network_arch_class_name,
            configuration.network_arch_init_kwargs,
            configuration.network_arch_init_kwargs_req_import,
            n_channels,
            output_channels=2,
            allow_init=True,
            deep_supervision=False
        ).to(device)

        checkpoint = torch.load(checkpoint_filename)

        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, weight_decay=3e-5, momentum=0.99)  # following nnunet
        optimizer.load_state_dict(checkpoint['optimizer'])

        if not args.wandb_ignore:
            wandb_run_id = checkpoint['wandb_run_id']
            wandb.init(project=args.wandb_project, mode="online", name=args.name, resume="must", id=wandb_run_id)

        # Initialize scheduler
        lr_scheduler = PolyLRScheduler(optimizer, args.learning_rate, args.n_epochs)  # following nnunet
        lr_scheduler.load_state_dict(checkpoint["scheduler"])

        print(f"\nResuming training: (epoch {checkpoint['epoch']})\nLoaded checkpoint '{checkpoint_filename}'\n")

    else:
        print(f"Initializing new model with {n_channels} input channels")
        model = get_network_from_plans(
            configuration.network_arch_class_name,
            configuration.network_arch_init_kwargs,
            configuration.network_arch_init_kwargs_req_import,
            n_channels,
            output_channels=2,
            allow_init=True,
            deep_supervision=False
        )

        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, weight_decay=3e-5, momentum=0.99)  # following nnunet

        start_epoch = 0
        if not args.wandb_ignore:
            wandb.login()
            wandb.init(project=args.wandb_project, mode="online", name=args.name)
            wandb_run_id = wandb.run.id

        # Initialize scheduler
        lr_scheduler = PolyLRScheduler(optimizer, args.learning_rate, args.n_epochs)  # following nnunet

    return model, optimizer, lr_scheduler, start_epoch, wandb_run_id


def main(args):
    seed_val = args.seed
    seed_everything(seed_val)

    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    dataset_name = convert_id_to_dataset_name(args.dataset_id)
    plans_file = pjoin(nnUNet_preprocessed, dataset_name, 'nnUNetPlans.json')
    plans_manager = PlansManagerInstanceSeg(plans_file)
    configuration = plans_manager.get_configuration('3d_fullres')
    n_channels = len(plans_manager.foreground_intensity_properties_per_channel)

    if args.debug:
        args.name = f"DEBUG_{args.name}"

    save_dir = pjoin(nnUNet_results, dataset_name, args.name, 'fold_%d' % args.fold)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.save_predictions or args.debug:
        patches_save_dir = pjoin(save_dir, 'saved_patches')
        os.makedirs(patches_save_dir, exist_ok=True)

    checkpoint_filename = os.path.join(save_dir, f"checkpoint_final.pth")
    model, optimizer, lr_scheduler, start_epoch, wandb_run_id = \
        get_model_optimizer_and_scheduler(args, configuration, n_channels, device, checkpoint_filename)

    # Initialize dataloaders
    train_loader = get_train_dataloader_from_dataset_id_and_fold(args.dataset_id, args.fold,
                                                                 num_workers=args.num_workers,
                                                                 cache_rate=args.cache_rate,
                                                                 seed_val=args.seed)

    if args.debug:
        # modify train_loader so that it only returns the first batch of train_loader
        val_loader = [next(iter(train_loader))]
        train_loader = val_loader * len(train_loader)
    else:
        val_loader = get_val_dataloader_from_dataset_id_and_fold(args.dataset_id, args.fold,
                                                                 num_workers=args.num_workers,
                                                                 cache_rate=args.cache_rate,
                                                                 seed_val=args.seed)

    # Initialize other variables and metrics
    epoch_num = args.n_epochs
    val_interval = args.val_interval
    best_val_loss = np.inf
    epoch_loss_values, metric_values_nDSC, metric_values_DSC = [], [], []
    best_dsc = -np.inf
    best_ndsc = -np.inf

    # Initialize scaler
    scaler = torch.cuda.amp.GradScaler()

    ''' Training loop '''
    for epoch in range(start_epoch, epoch_num):
        start_epoch_time = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = epoch_loss_ce = epoch_loss_dice = 0

        for batch_idx, batch_data in enumerate(train_loader):
            start_batch_time = time.time()
            inputs, labels = (
                batch_data["img"].to(device),
                batch_data["seg"].type(torch.LongTensor).to(device),
            )

            if (args.debug or args.save_predictions) and epoch == 0 and batch_idx == 0:
                save_patch(inputs.cpu().numpy(), f'{epoch}_image', patches_save_dir)
                save_patch(labels.cpu().numpy().astype(np.int16), f'{epoch}_labels', patches_save_dir)

            semantic_pred = model(inputs)

            if args.debug and (epoch+1) % val_interval == 0 and batch_idx == len(train_loader) - 1:
                save_patch(semantic_pred.detach().cpu().numpy().astype(np.int16), f'{epoch}_trainpred_labels', patches_save_dir)

            loss, dice_loss, focal_loss = compute_loss(semantic_pred, labels)

            epoch_loss += loss.item()
            epoch_loss_ce += focal_loss.item()
            epoch_loss_dice += dice_loss.item()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            elapsed_time = time.time() - start_batch_time
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)}, train_loss: {loss.item():.4f} " 
                f"(elapsed time: {int(elapsed_time // 60)}min {int(elapsed_time % 60)}s)")

        elapsed_epoch_time = time.time() - start_epoch_time
        print(f"Epoch {epoch + 1} took {int(elapsed_epoch_time // 60)}min {int(elapsed_epoch_time % 60)}s")

        epoch_loss /= len(train_loader)
        epoch_loss_dice /= len(train_loader)
        epoch_loss_ce /= len(train_loader)
        epoch_loss_values.append(epoch_loss)

        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()

        if not args.wandb_ignore:
            wandb.log(
                {'Training Loss/Total Loss': epoch_loss, 'Training Loss/Dice Loss': epoch_loss_dice,
                 'Training Loss/Focal Loss': epoch_loss_ce, 'Learning rate': current_lr},
                step=epoch)

        ##### Validation #####
        if (epoch + 1) % val_interval == 0:
            start_validation_time = time.time()
            model.eval()
            with torch.no_grad():
                avg_val_loss = avg_val_dice_loss = avg_val_ce_loss = 0
                total_dice = total_ndsc = 0
                batch_size = 0

                for batch_idx, val_data in enumerate(val_loader):
                    val_inputs, val_labels = (
                        val_data["img"].to(device),
                        val_data["seg"].type(torch.LongTensor).to(device),
                    )
                    val_semantic_pred, val_center_pred, val_offsets_pred = model(val_inputs)

                    act = torch.nn.Softmax(dim=1)
                    val_seg_pred = act(val_semantic_pred.clone().detach()).cpu().numpy()
                    val_seg_pred = val_seg_pred[:, 1, :, :, :]
                    torch.cuda.empty_cache()
                    binary_seg = np.copy(val_seg_pred)
                    binary_seg[binary_seg >= 0.5] = 1
                    binary_seg[binary_seg < 0.5] = 0
                    if len(binary_seg.shape) > 3:
                       binary_seg = np.squeeze(binary_seg)

                    if (args.debug or args.save_predictions) and batch_idx == 0:
                        save_patch(val_inputs.cpu().numpy(), f'{epoch}_pred_image', patches_save_dir)
                        save_patch(val_seg_pred[0], f"{epoch}_pred_segmentation_proba", patches_save_dir)
                        save_patch(binary_seg[0], f"{epoch}_pred_segmentation_binary", patches_save_dir)

                    batch_size = val_labels.shape[0]
                    for mbidx in range(batch_size):
                        total_dice += dice_metric(np.squeeze(val_labels.cpu().numpy()[mbidx]), binary_seg[mbidx])
                        total_ndsc += dice_norm_metric(np.squeeze(val_labels.cpu().numpy()[mbidx]), binary_seg[mbidx])

                    val_loss, val_dice_loss, val_focal_loss = compute_loss(val_semantic_pred, val_labels)

                    avg_val_loss += val_loss.item()
                    avg_val_dice_loss += val_dice_loss.item()
                    avg_val_ce_loss += val_focal_loss.item()

                avg_val_loss /= len(val_loader)
                avg_val_dice_loss /= len(val_loader)
                avg_val_ce_loss /= len(val_loader)
                total_dice /= (len(val_loader) * batch_size)
                total_ndsc /= (len(val_loader) * batch_size)

                if not args.wandb_ignore:
                    wandb.log(
                        {'Validation Loss/Total Loss': avg_val_loss, 'Validation Loss/Dice Loss': avg_val_dice_loss,
                         'Validation Segmentation Loss/Focal Loss': avg_val_ce_loss,
                         'Validation Metrics/Dice Score': total_dice,
                         'Validation Metrics/Normalized Dice': total_ndsc},
                        step=epoch
                    )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_path = os.path.join(save_dir, f"seed{args.seed}_best_loss.pth")
                    torch.save(model.state_dict(), save_path)

                if total_ndsc > best_ndsc:
                    best_ndsc = total_ndsc
                    save_path = os.path.join(save_dir, f"seed{args.seed}_best_ndsc.pth")
                    torch.save(model.state_dict(), save_path)

                if total_dice > best_dsc:
                    best_dsc = total_dice
                    save_path = os.path.join(save_dir, f"seed{args.seed}_best_dsc.pth")
                    torch.save(model.state_dict(), save_path)

                val_elapsed_time = time.time() - start_validation_time
                print(f"Validation took {int(val_elapsed_time // 60)}min {int(val_elapsed_time % 60)}s")
                print(f"Validation Loss: {avg_val_loss:.4f}")
                print(f"Validation DSC: {total_dice:.4f} (best DSC: {best_dsc:.4f})")
                print(f"Validation nDSC: {total_ndsc:.4f} (best nDSC: {best_ndsc:.4f})")

        if not args.debug and not args.wandb_ignore:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'wandb_run_id': wandb_run_id,
                'scheduler': lr_scheduler.state_dict()
            }, checkpoint_filename)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

