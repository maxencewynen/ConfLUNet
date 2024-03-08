import argparse
import os
import torch
from torch import nn
from monai.data import decollate_batch
from monai.transforms import Compose, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
import numpy as np
import random
from metrics import *
from data_load import get_train_dataloader, get_val_dataloader
import wandb
from os.path import join as pjoin
from metrics import *
from model import *
import time
from postprocess import postprocess
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Get all command line arguments.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# trainining
parser.add_argument('--separate_decoders', action="store_true", default=False,
                    help="Whether to use separate decoders for the segmentation and center prediction tasks")
parser.add_argument('--frozen_learning_rate', type=float, default=-1, help='Specify the initial learning rate')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify the initial learning rate')
parser.add_argument('--seg_loss_weight', type=float, default=1, help='Specify the weight of the segmentation loss')
parser.add_argument('--heatmap_loss_weight', type=float, default=100, help='Specify the weight of the heatmap loss')
parser.add_argument('--offsets_loss_weight', type=float, default=10, help='Specify the weight of the offsets loss')
parser.add_argument('--offsets_scale', type=float, default=1,
                    help='Specify the scale to multiply the predicted offsets with')
parser.add_argument('--offsets_loss', type=str, default="l1",
                    help="Specify the loss used for the offsets. ('sl1' or 'l1')")
parser.add_argument('--n_epochs', type=int, default=300, help='Specify the number of epochs to train for')
parser.add_argument('--path_model', type=str, default=None, help='Path to pretrained model')
# initialisation
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
# data
parser.add_argument('--data_dir', type=str, default=os.environ["DATA_ROOT_DIR"],
                    help='Specify the path to the data files directory')

parser.add_argument('--I', nargs='+', default=['FLAIR'], choices=['FLAIR', 'phase_T2starw', 'mag_T2starw', \
                                                                  'MPRAGE_reg-T2starw_T1w', 'T1map', 'UNIT1'])

parser.add_argument('--apply_mask', type=str, default=None,
                    help='The name of the folder containing the masks you want to the images to be applied to.')

parser.add_argument('--save_path', type=str, default=os.environ["MODELS_ROOT_DIR"],
                    help='Specify the path to the save directory')

parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--cache_rate', default=1.0, type=float)
# logging
parser.add_argument('--val_interval', type=int, default=5, help='Validation every n-th epochs')
parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold')
parser.add_argument('--minimum_lesion_size', type=int, default=14, help='Minimum lesion size in mm^3')

parser.add_argument('--wandb_project', type=str, default='WMLIS', help='wandb project name')
parser.add_argument('--name', default="idiot without a name", help='Wandb run name')
parser.add_argument('--force_restart', default=False, action='store_true',
                    help="force the training to restart at 0 even if a checkpoint was found")

VAL_AMP = True
roi_size = (96, 96, 96)


def load_checkpoint(model, optimizer, filename):
    start_epoch = 0
    print(f"=> Loading checkpoint '{filename}'")
    checkpoint = torch.load(filename)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    wandb_run_id = checkpoint['wandb_run_id']
    print(f"\nResuming training: (epoch {checkpoint['epoch']})\nLoaded checkpoint '{filename}'\n")

    return model, optimizer, start_epoch, wandb_run_id


def check_paths(args):
    from os.path import exists as pexists
    assert pexists(args.data_dir), f"Directory not found {args.data_dir}"
    # assert pexists(args.bm_path), f"Directory not found {args.bm_path}"
    assert pexists(args.save_path), f"Directory not found {args.save_path}"
    if args.path_model:
        assert pexists(args.path_model), f"Warning: File not found {args.path_model}"


def inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=roi_size,
            sw_batch_size=2,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        return torch.device('cuda')
    raise Exception("No GPU device was found: I cannot train without GPU.")


post_trans = Compose(
    [AsDiscrete(argmax=True, to_onehot=2)]
)


def main(args):
    check_paths(args)
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    set_determinism(seed=seed_val)

    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    save_dir = f'{args.save_path}/{args.name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    flr = args.learning_rate if args.frozen_learning_rate < 0 else args.frozen_learning_rate

    # Initialize model
    checkpoint_filename = os.path.join(save_dir, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_filename) and not args.force_restart:
        model = ConfLUNet(in_channels=len(args.I), num_classes=2, separate_decoders=args.separate_decoders,
                                  scale_offsets=args.offsets_scale).to(device)

        checkpoint = torch.load(checkpoint_filename)

        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        first_layer_params = model.a_block1.conv1.parameters()
        rest_of_model_params = [p for p in model.parameters() if p not in first_layer_params]

        optimizer = torch.optim.Adam([{'params': first_layer_params, 'lr': args.learning_rate},
                                      {'params': rest_of_model_params, 'lr': flr}],
                                     weight_decay=0.0005)  # momentum=0.9,
        optimizer.load_state_dict(checkpoint['optimizer'])

        wandb_run_id = checkpoint['wandb_run_id']
        wandb.init(project=args.wandb_project, mode="online", name=args.name, resume="must", id=wandb_run_id)

        # Initialize scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        lr_scheduler.load_state_dict(checkpoint["scheduler"])

        print(f"\nResuming training: (epoch {checkpoint['epoch']})\nLoaded checkpoint '{checkpoint_filename}'\n")

    else:
        if args.path_model is not None:
            print(f"Retrieving pretrained model from {args.path_model}")
            model = get_pretrained_model(args.path_model, len(args.I))
        else:
            print(f"Initializing new model with {len(args.I)} input channels")
            model = ConfLUNet(in_channels=len(args.I), num_classes=2, separate_decoders=args.separate_decoders,
                                      scale_offsets=args.offsets_scale).to(device)

        model.to(device)
        first_layer_params = model.a_block1.conv1.parameters()
        rest_of_model_params = [p for p in model.parameters() if p not in first_layer_params]
        optimizer = torch.optim.Adam([{'params': first_layer_params, 'lr': args.learning_rate},
                                      {'params': rest_of_model_params, 'lr': flr}],
                                     weight_decay=0.0005)  # momentum=0.9,

        start_epoch = 0
        wandb.login()
        wandb.init(project=args.wandb_project, mode="online", name=args.name)
        wandb_run_id = wandb.run.id

        # Initialize scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    # Initialize dataloaders
    train_loader = get_train_dataloader(data_dir=args.data_dir,
                                        num_workers=args.num_workers,
                                        I=args.I,
                                        cache_rate=args.cache_rate,
                                        apply_mask=args.apply_mask)
    val_loader = get_val_dataloader(data_dir=args.data_dir,
                                    num_workers=args.num_workers,
                                    I=args.I,
                                    cache_rate=args.cache_rate,
                                    apply_mask=args.apply_mask)

    # Initialize losses
    loss_function_dice = DiceLoss(to_onehot_y=True,
                                  softmax=True, sigmoid=False,
                                  include_background=False)
    loss_function_mse = nn.MSELoss()
    if args.offsets_loss == 'l1':
        offset_loss_fn = nn.L1Loss(reduction='none')
    elif args.offsets_loss == 'sl1':
        offset_loss_fn = nn.SmoothL1Loss(reduction='none')

    # Initialize other variables and metrics
    act = nn.Softmax(dim=1)
    epoch_num = args.n_epochs
    val_interval = args.val_interval
    threshold = args.threshold
    gamma_focal = 2.0
    dice_weight = 0.5
    focal_weight = 1.0
    seg_loss_weight = args.seg_loss_weight
    heatmap_loss_weight = args.heatmap_loss_weight
    offsets_loss_weight = args.offsets_loss_weight
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    best_metric_nDSC, best_metric_epoch_nDSC = -1, -1
    best_metric_DSC, best_metric_epoch_DSC = -1, -1
    best_metric_PQ, best_metric_epoch_PQ = -1, -1
    best_metric_F1, best_metric_epoch_F1 = -1, -1
    best_metric_mMV, best_metric_epoch_mMV = -1, -1
    epoch_loss_values, metric_values_nDSC, metric_values_DSC = [], [], []

    # Initialize scaler
    scaler = torch.cuda.amp.GradScaler()
    step_print = 1

    ''' Training loop '''
    for epoch in range(start_epoch, epoch_num):
        start_epoch_time = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        epoch_loss_ce = 0
        epoch_loss_dice = 0
        epoch_loss_seg = 0
        epoch_loss_mse = 0
        epoch_loss_offsets = 0
        step = 0

        for batch_data in train_loader:
            n_samples = batch_data["image"].size(0)
            for m in range(0, batch_data["image"].size(0), args.batch_size):
                step += args.batch_size
                inputs, labels, center_heatmap, offsets = (
                    batch_data["image"][m:(m + 2)].to(device),
                    batch_data["label"][m:(m + 2)].type(torch.LongTensor).to(device),
                    batch_data["center_heatmap"][m:(m + 2)].to(device),
                    batch_data["offsets"][m:(m + 2)].to(device))

                with torch.cuda.amp.autocast():
                    semantic_pred, center_pred, offsets_pred = model(inputs)

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

                    ### COM PREDICTION LOSS ###
                    mse_loss = loss_function_mse(center_pred, center_heatmap)

                    ### COM REGRESSION LOSS ###
                    # Disregard voxels outside the GT segmentation
                    offset_loss_weights_matrix = labels.expand_as(offsets_pred)
                    offset_loss = offset_loss_fn(offsets_pred, offsets) * offset_loss_weights_matrix
                    if offset_loss_weights_matrix.sum() > 0:
                        offset_loss = offset_loss.sum() / offset_loss_weights_matrix.sum()
                    else:  # No foreground voxels
                        offset_loss = offset_loss.sum() * 0

                    ### TOTAL LOSS ###
                    loss = (seg_loss_weight * segmentation_loss) + (heatmap_loss_weight * mse_loss) + (
                            offsets_loss_weight * offset_loss)

                epoch_loss += loss.item()
                epoch_loss_ce += focal_loss.item()
                epoch_loss_dice += dice_loss.item()
                epoch_loss_seg += segmentation_loss.item()
                epoch_loss_mse += mse_loss.item()
                epoch_loss_offsets += offset_loss.item()

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

                if step % 100 == 0:
                    elapsed_time = time.time() - start_epoch_time
                    step_print = int(step / args.batch_size)
                    print(
                        f"{step_print}/{(len(train_loader) * n_samples) // (train_loader.batch_size * args.batch_size)}, train_loss: {loss.item():.4f}" + \
                        f"(elapsed time: {int(elapsed_time // 60)}min {int(elapsed_time % 60)}s)")

        epoch_loss /= step_print
        epoch_loss_dice /= step_print
        epoch_loss_ce /= step_print
        epoch_loss_seg /= step_print
        epoch_loss_mse /= step_print
        epoch_loss_offsets /= step_print
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()

        wandb.log(
            {'Training Loss/Total Loss': epoch_loss, 'Training Segmentation Loss/Dice Loss': epoch_loss_dice,
             'Training Segmentation Loss/Focal Loss': epoch_loss_ce,
             'Training Loss/Segmentation Loss': epoch_loss_seg, 'Training Loss/Center Prediction Loss': epoch_loss_mse,
             'Training Loss/Offsets Loss': epoch_loss_offsets, 'Learning rate': current_lr, },
            step=epoch)

        ##### Validation #####
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                nDSC_list = []
                for val_data in val_loader:
                    val_inputs, val_labels, val_heatmaps, val_offsets, val_bms = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                        val_data["center_heatmap"].to(device),
                        val_data["offsets"].to(device),
                        val_data["brain_mask"].squeeze().cpu().numpy()
                    )

                    vsp, val_center_pred, val_offsets_pred = inference(val_inputs, model)

                    for_dice_outputs = [post_trans(i) for i in decollate_batch(vsp)]

                    dice_metric(y_pred=for_dice_outputs, y=val_labels)
                    val_semantic_pred = act(vsp)[:, 1]
                    val_semantic_pred = torch.where(val_semantic_pred >= threshold, torch.tensor(1.0).to(device),
                                                    torch.tensor(0.0).to(device))
                    val_semantic_pred = val_semantic_pred.squeeze().cpu().numpy()
                    nDSC = dice_norm_metric(val_labels.squeeze().cpu().numpy()[val_bms == 1],
                                            val_semantic_pred[val_bms == 1])
                    nDSC_list.append(nDSC)

                del val_inputs, val_labels, val_semantic_pred, val_heatmaps, val_offsets, for_dice_outputs, val_bms  # , thresholded_output, curr_preds, gts , val_bms
                torch.cuda.empty_cache()
                metric_nDSC = np.mean(nDSC_list)
                metric_DSC = dice_metric.aggregate().item()
                wandb.log({
                    'Segmentation Metrics/nDSC (val)': metric_nDSC,
                    'Segmentation Metrics/ DSC (val)': metric_DSC, }, step=epoch)

                metric_values_nDSC.append(metric_nDSC)
                metric_values_DSC.append(metric_DSC)

                metric_PQ, metric_F1, metric_recall, metric_precision, metric_dic, metric_dice_per_tp, metric_mMV = 0, 0, 0, 0, 0, 0, 0

                if metric_DSC > 0.6:
                    pqs = []
                    fbetas = []
                    recalls = []
                    precisions = []
                    dics = []
                    dice_scores_per_tp = []
                    max_votes = []

                    print(f"Computing instance segmentation metrics! (DSC = {metric_DSC:.4f})")
                    for val_data in tqdm(val_loader):

                        val_inputs, val_instances, val_labels, val_heatmaps, val_offsets, val_bms = (
                            val_data["image"].to(device),
                            val_data["instance_mask"].to(device),
                            val_data["label"].to(device),
                            val_data["center_heatmap"].to(device),
                            val_data["offsets"].to(device),
                            val_data["brain_mask"].squeeze().cpu().numpy()
                        )

                        meta_dict = args.I[0] + "_meta_dict"
                        voxel_size = tuple(val_data[meta_dict]['pixdim'][0][1:4])

                        val_semantic_pred, val_center_pred, val_offsets_pred = inference(val_inputs, model)

                        val_semantic_pred = act(val_semantic_pred).cpu().numpy()
                        val_semantic_pred = np.squeeze(val_semantic_pred[0, 1]) * val_bms
                        del val_inputs
                        torch.cuda.empty_cache()

                        seg = val_semantic_pred
                        seg[seg >= args.threshold] = 1
                        seg[seg < args.threshold] = 0
                        seg = np.squeeze(seg)

                        sem_pred, inst_pred, _, votes = postprocess(val_semantic_pred,
                                                                    val_center_pred,
                                                                    val_offsets_pred,
                                                                    compute_voting=True,
                                                                    voxel_size=voxel_size, )
                        votes *= val_bms

                        matched_pairs, unmatched_pred, unmatched_ref = match_instances(inst_pred,
                                                                                       val_instances.squeeze().cpu().numpy())
                        pq_val = panoptic_quality(pred=inst_pred, ref=val_instances.squeeze().cpu().numpy(),
                                                  matched_pairs=matched_pairs, unmatched_pred=unmatched_pred,
                                                  unmatched_ref=unmatched_ref)

                        fbeta_val = f_beta_score(matched_pairs=matched_pairs, unmatched_pred=unmatched_pred,
                                                 unmatched_ref=unmatched_ref)
                        recall_val = recall(matched_pairs=matched_pairs, unmatched_ref=unmatched_ref)
                        precision_val = precision(matched_pairs=matched_pairs, unmatched_pred=unmatched_pred)
                        dice_scores = dice_per_tp(inst_pred, val_instances.squeeze().cpu().numpy(), matched_pairs)
                        avg_dice_scores = sum(dice_scores) / len(dice_scores) if dice_scores else 0
                        dic = DiC(inst_pred, val_instances.squeeze().cpu().numpy())

                        pqs += [pq_val]
                        fbetas += [fbeta_val]
                        recalls += [recall_val]
                        precisions += [precision_val]
                        dics += [dic]
                        max_votes += [votes.max()]
                        if avg_dice_scores > 0:
                            dice_scores_per_tp += [avg_dice_scores]

                    metric_PQ = np.mean(pqs)
                    metric_F1 = np.mean(fbetas)
                    metric_ltpr = np.mean(ltprs)
                    metric_precision = np.mean(precisions)
                    metric_dic = np.mean(dics)
                    metric_dice_per_tp = np.mean(dice_scores_per_tp)
                    metric_mMV = np.mean(max_votes)

                    wandb.log({'Instance Segmentation Metrics/PQ (val)': metric_PQ,
                               'Instance Segmentation Metrics/F1 (val)': metric_F1,
                               'Instance Segmentation Metrics/LTPR (val)': metric_ltpr,
                               'Instance Segmentation Metrics/precision (val)': metric_precision,
                               'Instance Segmentation Metrics/DIC (val)': metric_dic,
                               'Instance Segmentation Metrics/Dice per TP (val)': metric_dice_per_tp,
                               'Offsets Metrics/Mean max vote (val)': metric_mMV
                               }, step=epoch)

                if metric_nDSC > best_metric_nDSC and epoch > 5:
                    best_metric_nDSC = metric_nDSC
                    best_metric_epoch_nDSC = epoch + 1
                    save_path = os.path.join(save_dir, f"best_nDSC_{args.name}_seed{args.seed}.pth")
                    torch.save(model.state_dict(), save_path)
                    print("saved new best metric model for nDSC")

                if metric_DSC > best_metric_DSC and epoch > 5:
                    best_metric_DSC = metric_DSC
                    best_metric_epoch_DSC = epoch + 1
                    save_path = os.path.join(save_dir, f"best_DSC_{args.name}_seed{args.seed}.pth")
                    torch.save(model.state_dict(), save_path)
                    print("saved new best metric model for DSC")

                if metric_PQ > best_metric_PQ and epoch > 5:
                    best_metric_PQ = metric_PQ
                    best_metric_epoch_PQ = epoch + 1
                    save_path = os.path.join(save_dir, f"best_PQ_{args.name}_seed{args.seed}.pth")
                    torch.save(model.state_dict(), save_path)
                    print("saved new best metric model for PQ")

                if metric_F1 > best_metric_F1 and epoch > 5:
                    best_metric_F1 = metric_F1
                    best_metric_epoch_F1 = epoch + 1
                    save_path = os.path.join(save_dir, f"best_F1_{args.name}_seed{args.seed}.pth")
                    torch.save(model.state_dict(), save_path)
                    print("saved new best metric model for F1")

                if metric_mMV > best_metric_mMV and epoch > 5:
                    best_metric_mMV = metric_F1
                    best_metric_epoch_mMV = epoch + 1
                    save_path = os.path.join(save_dir, f"best_mMV_{args.name}_seed{args.seed}.pth")
                    torch.save(model.state_dict(), save_path)
                    print("saved new best metric model for mean max votes")

                print(f"current epoch: {epoch + 1} current mean normalized dice: {metric_nDSC:.4f}"
                      f"\nbest mean normalized dice: {best_metric_nDSC:.4f} at epoch: {best_metric_epoch_nDSC}"
                      f"\nbest mean dice: {best_metric_DSC:.4f} at epoch: {best_metric_epoch_DSC}"
                      f"\nbest mean PQ: {best_metric_PQ:.4f} at epoch: {best_metric_epoch_PQ}"
                      f"\nbest mean F1: {best_metric_F1:.4f} at epoch: {best_metric_epoch_F1}"
                      )

                dice_metric.reset()

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