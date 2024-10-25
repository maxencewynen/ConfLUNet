import os
import argparse
from os.path import join as pjoin
import nibabel as nib
import numpy as np

from nnunetv2.paths import nnUNet_preprocessed

from conflunet.postprocessing.small_instances_removal import remove_small_lesions_from_instance_segmentation
from conflunet.utilities.planning_and_configuration import load_dataset_and_configuration
from conflunet.evaluation.metrics import *


def compute_metrics_from_preprocessed_data_model_postprocessor_and_fold(
        dataset_id: str,
        output_dir: str,
        model_name: str,
        postprocessor_name: str,
        fold: int,
        verbose: bool = False
):
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(dataset_id)

    predictions_dir = pjoin(output_dir, dataset_name, model_name, f"fold_{fold}", postprocessor_name)
    reference_dir = pjoin(nnUNet_preprocessed, dataset_name, configuration.data_identifier)
    all_metrics = {k: [] for k in METRICS_TO_AVERAGE}
    all_metrics.update({k: [] for k in METRICS_TO_SUM})

    for filename in sorted(os.listdir(predictions_dir)):
        print(filename)
        if os.path.isdir(pjoin(predictions_dir, filename)) or not filename.endswith('.nii.gz'):
            continue
        if "instance_seg_pred" not in filename:
            continue
        case_identifier = filename.replace(f'_instance_seg_pred_{postprocessor_name}.nii.gz', '')

        instance_seg_pred = nib.load(pjoin(predictions_dir, filename)).get_fdata()
        ref_data = np.squeeze(np.load(pjoin(reference_dir, f'{case_identifier}.npz'))['instance_seg'])
        instance_seg_ref = remove_small_lesions_from_instance_segmentation(ref_data, voxel_size=configuration.spacing)
        metrics, all_pred_matches, all_ref_matches = compute_metrics(
            instance_seg_pred,
            instance_seg_ref,
            voxel_size=configuration.spacing,
            verbose=verbose
        )
        for k in metrics.keys():
            all_metrics[k].append(metrics[k])
        break

    for k in all_metrics.keys():
        all_metrics[k] = np.array(all_metrics[k])
        if k in METRICS_TO_AVERAGE:
            all_metrics[k] = np.mean(all_metrics[k])
        elif k in METRICS_TO_SUM:
            all_metrics[k] = np.sum(all_metrics[k])

    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation from model name and postprocessor name')
    # parser.add_argument('--dataset_name', type=str)
    # parser.add_argument('--model_name', type=str)
    # parser.add_argument('--postprocessor_name', type=str)
    # parser.add_argument('--save_dir', type=str, help='Path to the output directory')
    #
    # args = parser.parse_args()

    compute_metrics_from_preprocessed_data_model_postprocessor_and_fold(
        321,
        None,
        None,
        "ConfLUNet",
        None,
        verbose=True
    )