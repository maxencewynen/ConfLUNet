import os
import numpy as np
from torch import nn
from os.path import join as pjoin
from typing import Dict, List, Union, Optional

from monai.data import DataLoader
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed

from conflunet.dataloading.dataloaders import get_full_val_dataloader_from_dataset_id_and_fold
from conflunet.inference.predictors.instance import ConfLUNetPredictor
from conflunet.inference.predictors.semantic import SemanticPredictor
from conflunet.postprocessing.instance import ConfLUNetPostprocessor
from conflunet.postprocessing.semantic import ACLSPostprocessor, ConnectedComponentsPostprocessor
from conflunet.utilities.planning_and_configuration import load_dataset_and_configuration, \
    ConfigurationManagerInstanceSeg, PlansManagerInstanceSeg
from conflunet.architecture.utils import load_model
from conflunet.training.utils import get_default_device
from conflunet.evaluation.metrics import *
from conflunet.evaluation.utils import *

POSTPROCESSORS = {
    "ConfLUNet": ConfLUNetPostprocessor,
    "ACLS": ACLSPostprocessor,
    "ConnectedComponents": ConnectedComponentsPostprocessor
}


def predict_and_evaluate(
        predictor: Union[ConfLUNetPredictor, SemanticPredictor],
        full_val_loader: DataLoader,
        model: nn.Module,
        configuration: ConfigurationManagerInstanceSeg,
        save_dir: str,
        num_workers: int = 4,
        do_i_compute_metrics: bool = True,
        verbose: bool = True
):
    # Predict
    all_metrics = {}
    all_pred_matches = {}
    all_ref_matches = {}
    predictions_loader = predictor.get_predictions_loader(full_val_loader, model, num_workers)
    for data_batch, predicted_batch in zip(full_val_loader, predictions_loader):
        instance_seg_pred = np.squeeze(predicted_batch['instance_seg_pred'].detach().cpu().numpy())
        instance_seg_ref = np.squeeze(data_batch['instance_seg'].detach().cpu().numpy())
        if do_i_compute_metrics:
            metrics, pred_matches, ref_matches = compute_metrics(
                instance_seg_pred,
                instance_seg_ref,
                voxel_size=configuration.spacing,
                verbose=verbose
            )
            name = data_batch['name'][0]
            all_metrics[name] = metrics
            all_pred_matches[name] = pred_matches
            all_ref_matches[name] = ref_matches

        # Save predictions
        predictor.save_predictions(predicted_batch)

    # Save metrics
    if do_i_compute_metrics:
        save_metrics(all_metrics, all_pred_matches, all_ref_matches, save_dir)


def predict_fold(
        dataset_id: int,
        fold: int,
        model_name: str,
        save_dir: str,
        num_workers: int = 4,
        postprocessor: Union[str, ACLSPostprocessor, ConnectedComponentsPostprocessor] = "ConfLUNet",
        dataset_name: str = None,
        plans_manager: PlansManagerInstanceSeg = None,
        configuration: ConfigurationManagerInstanceSeg = None,
        n_channels: int = None,
        device: str = None,
        save_only_instance_segmentation: bool = False,
        convert_to_original_shape: bool = False,
        do_i_compute_metrics: bool = True,
        verbose: bool = True
):
    if dataset_name is None or plans_manager is None or configuration is None or n_channels is None:
        dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(dataset_id)
    if device is None:
        device = get_default_device()

    # Get dataloader
    full_val_loader = get_full_val_dataloader_from_dataset_id_and_fold(dataset_id, fold, num_workers)

    # Load model
    if postprocessor == "ConfLUNet":
        postprocessor_name = "ConfLUNet"
        metric = "Panoptic_Quality"
        checkpoint_filename = f"checkpoint_best_{metric}_{postprocessor_name}.pth"
        semantic = False
    else:
        postprocessor_name = postprocessor.name
        metric = "Dice_Score"
        checkpoint_filename = f"checkpoint_best_{metric}_{postprocessor_name}.pth"
        semantic = True

    checkpoint = pjoin(nnUNet_results, dataset_name, model_name, f"fold_{fold}", checkpoint_filename)
    model = load_model(configuration, checkpoint, n_channels, semantic=semantic)
    model.to(device)
    model.eval()

    # Set paths
    assert os.path.exists(save_dir), f"Path {save_dir} does not exist"
    save_dir = pjoin(save_dir, dataset_name, model_name, f"fold_{fold}", postprocessor_name)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize predictor based on postprocessor type
    if postprocessor == "ConfLUNet":
        predictor = ConfLUNetPredictor(
            plans_manager=plans_manager,
            model=model,
            postprocessor=ConfLUNetPostprocessor(
                minimum_instance_size=14,
                minimum_size_along_axis=3,
                voxel_spacing=configuration.spacing,
                semantic_threshold=0.5,
                heatmap_threshold=0.1,
                nms_kernel_size=3,
                top_k=150,
                compute_voting=False,
                calibrate_offsets=False,
                device=device,
                verbose=verbose
            ),
            output_dir=save_dir,
            num_workers=num_workers,
            convert_to_original_shape=convert_to_original_shape,
            save_only_instance_segmentation=save_only_instance_segmentation,
            verbose=verbose
        )
    else:
        predictor = SemanticPredictor(
            plans_manager=plans_manager,
            model=model,
            postprocessor=postprocessor,
            output_dir=save_dir,
            num_workers=num_workers,
            convert_to_original_shape=convert_to_original_shape,
            save_only_instance_segmentation=save_only_instance_segmentation,
            verbose=verbose
        )

    # Predict
    print(f"[INFO] Starting inference for fold {fold}...")
    predict_and_evaluate(
        predictor=predictor,
        full_val_loader=full_val_loader,
        model=model,
        configuration=configuration,
        save_dir=save_dir,
        num_workers=num_workers,
        do_i_compute_metrics=do_i_compute_metrics,
        verbose=verbose
    )
    print(f"[INFO] Inference for fold {fold} completed.")


def predict_all_folds(
        dataset_id: int,
        model_name: str,
        semantic: bool,
        save_dir: str,
        num_workers: int = 4,
        save_only_instance_segmentation: bool = False,
        convert_to_original_shape: bool = False,
        do_i_compute_metrics: bool = True,
        verbose: bool = True
):
    if "SEMANTIC" in model_name:
        semantic = True

    # Load dataset and configuration
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(dataset_id)

    device = get_default_device()

    if semantic:
        postprocessors = [
            ConnectedComponentsPostprocessor(
                minimum_instance_size=14,
                minimum_size_along_axis=3,
                voxel_spacing=configuration.spacing,
                semantic_threshold=0.5,
                device=device,
                verbose=False
            ),
            ACLSPostprocessor(
                minimum_instance_size=14,
                minimum_size_along_axis=3,
                voxel_spacing=configuration.spacing,
                semantic_threshold=0.5,
                sigma=1.0,
                device=device,
                verbose=False
            ),
        ]
    else:
        postprocessors = ["ConfLUNet"]

    for fold in range(5):
        for postprocessor in postprocessors:
            predict_fold(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                plans_manager=plans_manager,
                configuration=configuration,
                n_channels=n_channels,
                fold=fold,
                device=device,
                model_name=model_name,
                save_dir=save_dir,
                num_workers=num_workers,
                postprocessor=postprocessor,
                save_only_instance_segmentation=save_only_instance_segmentation,
                convert_to_original_shape=convert_to_original_shape,
                do_i_compute_metrics=do_i_compute_metrics,
                verbose=verbose
            )

    if do_i_compute_metrics:
        # summarize metrics
        postprocessor_names = []
        for p in postprocessors:
            if isinstance(p, str):
                postprocessor_names += [p]
            else:
                postprocessor_names += [p.name]

        for pp_name in postprocessor_names:
            summarize_metrics_from_model_and_postprocessor(dataset_name, model_name, pp_name, save_dir)



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict and evaluate folds.")
    parser.add_argument("--dataset_id", required=True, type=int, help="Dataset ID.")
    parser.add_argument("--model_name", required=True, type=str, help="Model name.")
    parser.add_argument("--semantic", action="store_true", help="Semantic segmentation model.")
    parser.add_argument("--save_dir", required=True, type=str, help="Path to save directory.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--save_only_instance_segmentation", action="store_true", help="Save only instance segmentation.")
    parser.add_argument("--convert_to_original_shape", action="store_true", help="Convert to original shape.")
    parser.add_argument("--do_not_compute_metrics", action="store_true", help="Don't compute metrics.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    parser.add_argument("--fold", type=int, help="Fold number.")
    args = parser.parse_args()
    if args.fold is not None:
        predict_fold(
            dataset_id=args.dataset_id,
            fold=args.fold,
            model_name=args.model_name,
            save_dir=args.save_dir,
            num_workers=args.num_workers,
            save_only_instance_segmentation=args.save_only_instance_segmentation,
            convert_to_original_shape=args.convert_to_original_shape,
            do_i_compute_metrics=not args.do_not_compute_metrics,
            verbose=args.verbose
        )
    else:
        predict_all_folds(
            dataset_id=args.dataset_id,
            model_name=args.model_name,
            semantic=args.semantic,
            save_dir=args.save_dir,
            num_workers=args.num_workers,
            save_only_instance_segmentation=args.save_only_instance_segmentation,
            convert_to_original_shape=args.convert_to_original_shape,
            do_i_compute_metrics=not args.do_not_compute_metrics,
            verbose=args.verbose
        )
    pass
