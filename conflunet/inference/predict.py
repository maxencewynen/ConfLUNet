import os
import json
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
    ConfigurationManagerInstanceSeg
from conflunet.architecture.utils import load_model
from conflunet.training.utils import get_default_device
from conflunet.evaluation.metrics import compute_metrics

POSTPROCESSORS = {
    "ConfLUNet": ConfLUNetPostprocessor,
    "ACLS": ACLSPostprocessor,
    "ConnectedComponents": ConnectedComponentsPostprocessor
}

METRICS_TO_AVERAGE = ["PQ", "DSC", "nDSC", "F1", "Recall", "Precision", "Dice_Per_TP", "DiC", "Recall_CLU", "Precision_CLU", "Dice_Per_TP_CLU"]
METRICS_TO_SUM = ["Pred_Lesion_Count", "Ref_Lesion_Count", "CLU_Count", "TP_CLU"]


def convert_types(obj):
    # Convert all np.int32 types to standard python int for json dumping
    if isinstance(obj, np.int32):
        return int(obj)
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_types(i) for i in obj]
    return obj


def save_metrics(
        all_metrics: Dict[str, Dict[str, float]],
        all_pred_matches: Dict[str, Dict[str, List[float]]],
        all_ref_matches: Dict[str, Dict[str, List[float]]],
        save_dir: str
) -> None:
    metrics_file = pjoin(save_dir, "metrics_details.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    pred_matches_file = pjoin(save_dir, "pred_matches.json")
    with open(pred_matches_file, 'w') as f:
        json.dump(convert_types(all_pred_matches), f, indent=4)

    ref_matches_file = pjoin(save_dir, "ref_matches.json")
    with open(ref_matches_file, 'w') as f:
        json.dump(convert_types(all_ref_matches), f, indent=4)

    # compute mean metrics or sum when applicable
    metrics_summary = {}
    metrics_summary.update({metric: np.nanmean([d[metric] for d in all_metrics.values()]) for metric in METRICS_TO_AVERAGE})
    metrics_summary.update({metric: np.nansum([d[metric] for d in all_metrics.values()]) for metric in METRICS_TO_SUM})

    metrics_summary_file = pjoin(save_dir, "metrics_summary.json")
    with open(metrics_summary_file, 'w') as f:
        json.dump(metrics_summary, f, indent=4)


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
            break

        # Save predictions
        predictor.save_predictions(predicted_batch)

    # Save metrics
    if do_i_compute_metrics:
        save_metrics(all_metrics, all_pred_matches, all_ref_matches, save_dir)


def predict_fold_ConfLUNet(
        dataset_id: int,
        fold: int,
        model_name: str,
        save_dir: str,
        num_workers: int = 4,
        save_only_instance_segmentation: bool = False,
        convert_to_original_shape: bool = False,
        do_i_compute_metrics: bool = True,
        verbose: bool = True
):
    # Load dataset and configuration
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(dataset_id)

    device = get_default_device()

    # Get dataloader
    full_val_loader = get_full_val_dataloader_from_dataset_id_and_fold(dataset_id, fold, num_workers)

    # Load model
    postprocessor_name = "ConfLUNet"
    metric = "Panoptic_Quality"
    filename = f"checkpoint_best_{metric}_{postprocessor_name}.pth"
    checkpoint = pjoin(nnUNet_results, dataset_name, model_name, f"fold_{fold}", filename)
    model = load_model(configuration, checkpoint, n_channels, False)
    model.to(device)
    model.eval()

    # Set paths
    assert os.path.exists(save_dir), f"Path {save_dir} does not exist"
    save_dir = pjoin(save_dir, model_name, f"fold_{fold}", postprocessor_name)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize predictor
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


def predict_fold_semantic(
        dataset_id: int,
        fold: int,
        model_name: str,
        postprocessor: Union[ACLSPostprocessor, ConnectedComponentsPostprocessor],
        save_dir: str,
        num_workers: int = 4,
        save_only_instance_segmentation: bool = False,
        convert_to_original_shape: bool = False,
        do_i_compute_metrics: bool = True,
        verbose: bool = True
):
    # Load dataset and configuration
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(dataset_id)

    device = get_default_device()

    # Get dataloader
    full_val_loader = get_full_val_dataloader_from_dataset_id_and_fold(dataset_id, fold, num_workers)

    # Load model
    postprocessor_name = postprocessor.name
    metric = "DSC"
    filename = f"checkpoint_best_{metric}_{postprocessor_name}.pth"
    checkpoint = pjoin(nnUNet_results, dataset_name, model_name, f"fold_{fold}", filename)
    model = load_model(configuration, checkpoint, n_channels, semantic=True)
    model.to(device)
    model.eval()

    # Set paths
    assert os.path.exists(save_dir), f"Path {save_dir} does not exist"
    save_dir = pjoin(save_dir, model_name, f"fold_{fold}", postprocessor_name)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize predictor
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
    for fold in range(5):
        predict_fold_ConfLUNet(
            dataset_id=dataset_id,
            fold=fold,
            model_name=model_name,
            save_dir=save_dir,
            num_workers=num_workers,
            save_only_instance_segmentation=save_only_instance_segmentation,
            convert_to_original_shape=convert_to_original_shape,
            do_i_compute_metrics=do_i_compute_metrics,
            verbose=verbose
        )

    if do_i_compute_metrics:
        # summarize metrics
        all_metrics = {}
        for fold in range(5):
            fold_metrics_file = pjoin(save_dir, model_name, f"fold_{fold}", "ConfLUNet", "metrics_summary.json")
            with open(fold_metrics_file, 'r') as f:
                this_fold_metrics = json.load(f)
                all_metrics[f"fold_{fold}"] = this_fold_metrics

        metrics_folds_details_file = pjoin(save_dir, model_name, "ConfLUNet", "metrics_fold_details.json")
        with open(metrics_folds_details_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)

        # compute mean metrics or sum when applicable
        metrics_summary = {}
        metrics_summary.update({metric: np.mean([d[metric] for d in all_metrics.values()]) for metric in METRICS_TO_AVERAGE})
        metrics_summary.update({metric: np.sum([d[metric] for d in all_metrics.values()]) for metric in METRICS_TO_SUM})

        metrics_summary_file = pjoin(save_dir, model_name, "ConfLUNet", "metrics_avg_across_folds.json")
        with open(metrics_summary_file, 'w') as f:
            json.dump(metrics_summary, f, indent=4)

        metrics_std = {}
        metrics_std.update({metric: np.std([d[metric] for d in all_metrics.values()]) for metric in METRICS_TO_AVERAGE})
        metrics_std_file = pjoin(save_dir, model_name, "ConfLUNet", "metrics_std_across_folds.json")
        with open(metrics_std_file, 'w') as f:
            json.dump(metrics_std, f, indent=4)


if __name__=="__main__":
    predict_fold_ConfLUNet(
        dataset_id=321,
        fold=0,
        model_name="lre-2_s1_o1-h1_fold0",
        save_dir="/home/mwynen/data/nnUNet_v2/nnUNet_output/Dataset321_WMLIS",
        num_workers=8,
        save_only_instance_segmentation=False,
        convert_to_original_shape=False,
        do_i_compute_metrics=True,
        verbose=True
    )
    pass
