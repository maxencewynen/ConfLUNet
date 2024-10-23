import os
import json
import numpy as np
from pprint import pprint
from os.path import join as pjoin
from typing import Dict, List, Union, Optional

from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed

from conflunet.dataloading.dataloaders import get_full_val_dataloader_from_dataset_id_and_fold
from conflunet.inference.predictors.instance import ConfLUNetPredictor
from conflunet.inference.predictors.semantic import SemanticPredictor
from conflunet.postprocessing.instance import ConfLUNetPostprocessor
from conflunet.postprocessing.semantic import ACLSPostprocessor, ConnectedComponentsPostprocessor
from conflunet.utilities.planning_and_configuration import load_dataset_and_configuration
from conflunet.architecture.utils import load_model
from conflunet.training.utils import get_default_device
from conflunet.evaluation.metrics import compute_metrics

POSTPROCESSORS = {
    "ConfLUNet": ConfLUNetPostprocessor,
    "ACLS": ACLSPostprocessor,
    "ConnectedComponents": ConnectedComponentsPostprocessor
}

METRICS_TO_AVERAGE = ["PQ", "DSC", "nDSC", "F1", "Recall", "Precision", "Dice_Per_TP", "DiC", "Recall_CLU", "Precision_CLU", "Dice_Per_TP_CLU"]
METRICS_TO_SUM = ["Pred_Lesion_Count", "Ref_Lesion_Count", "CLU_Count"]


def save_metrics(
        all_metrics: List[Dict[str, float]],
        all_pred_matches: List[Dict[str, List[float]]],
        all_ref_matches: List[Dict[str, List[float]]],
        save_dir: str
) -> None:
    metrics_file = pjoin(save_dir, "metrics_details.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)

    pred_matches_file = pjoin(save_dir, "pred_matches.json")
    with open(pred_matches_file, 'w') as f:
        json.dump(all_pred_matches, f, indent=4)

    ref_matches_file = pjoin(save_dir, "ref_matches.json")
    with open(ref_matches_file, 'w') as f:
        json.dump(all_ref_matches, f, indent=4)

    # compute mean metrics or sum when applicable
    metrics_summary = {}
    for metric in METRICS_TO_AVERAGE:
        metrics_summary[metric] = sum([m[metric] for m in all_metrics]) / len(all_metrics)
    for metric in METRICS_TO_SUM:
        metrics_summary[metric] = sum([m[metric] for m in all_metrics])
    if "Recall_CLU" in metrics_summary.keys() and "CLU_Count" in metrics_summary.keys():
        metrics_summary["TP_CLU"] = round(metrics_summary["Recall_CLU"] * metrics_summary['CLU_Count'])
    else:
        metrics_summary['TP_CLU'] = np.nan

    metrics_summary_file = pjoin(save_dir, "metrics_summary.json")
    with open(metrics_summary_file, 'w') as f:
        json.dump(metrics_summary, f, indent=4)


def predict_fold_ConfLUNet(
        dataset_id: int,
        fold: int,
        model_name: str,
        semantic: bool,
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
    model = load_model(configuration, checkpoint, n_channels, semantic)
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
    all_metrics = []
    all_pred_matches = []
    all_ref_matches = []
    predictions_loader = predictor.get_predictions_loader(full_val_loader, model, num_workers)
    print(f"[INFO] Starting inference for fold {fold}...")
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
            metrics['name'] = data_batch['name']
            pprint(metrics, width=1)
            all_metrics.append(metrics)

            all_pred_matches['name'] = data_batch['name']
            all_pred_matches.append(pred_matches)
            all_ref_matches['name'] = data_batch['name']
            all_ref_matches.append(ref_matches)

        # Save predictions
        predictor.save_predictions(predicted_batch)

    # Save metrics
    if do_i_compute_metrics:
        save_metrics(all_metrics, all_pred_matches, all_ref_matches, save_dir)


if __name__=="__main__":
    predict_fold_ConfLUNet(
        dataset_id=321,
        fold=0,
        model_name="lre-2_s1_o1-h1_fold0",
        semantic=False,
        save_dir="/home/mwynen/data/nnUNet_v2/nnUNet_output/Dataset321_WMLIS",
        num_workers=8,
        save_only_instance_segmentation=False,
        convert_to_original_shape=False,
        do_i_compute_metrics=True,
        verbose=True
    )
    pass
