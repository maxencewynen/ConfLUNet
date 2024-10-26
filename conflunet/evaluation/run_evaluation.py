import os
import nibabel as nib
import numpy as np

from nnunetv2.paths import nnUNet_preprocessed

from conflunet.postprocessing.small_instances_removal import remove_small_lesions_from_instance_segmentation
from conflunet.utilities.planning_and_configuration import load_dataset_and_configuration
from conflunet.evaluation.metrics import *
from conflunet.evaluation.utils import *


def compute_metrics_from_preprocessed_data_model_postprocessor_and_fold(
        dataset_id: str,
        output_dir: str,
        model_name: str,
        postprocessor_name: str,
        fold: int,
        verbose: bool = False
) -> Dict[str, float]:
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(dataset_id)

    predictions_dir = pjoin(output_dir, dataset_name, model_name, f"fold_{fold}", postprocessor_name)
    reference_dir = pjoin(nnUNet_preprocessed, dataset_name, configuration.data_identifier)

    all_metrics = {}
    all_pred_matches = {}
    all_ref_matches = {}

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
        metrics, pred_matches, ref_matches = compute_metrics(
            instance_seg_pred,
            instance_seg_ref,
            voxel_size=configuration.spacing,
            verbose=verbose
        )
        all_metrics[case_identifier] = metrics
        all_pred_matches[case_identifier] = pred_matches
        all_ref_matches[case_identifier] = ref_matches

    save_dir = pjoin(output_dir, dataset_name, model_name, f"fold_{fold}", postprocessor_name)
    save_metrics(all_metrics, all_pred_matches, all_ref_matches, save_dir)


def process_fold(fold, dataset_id, output_dir, model_name, postprocessor_name, verbose):
    return compute_metrics_from_preprocessed_data_model_postprocessor_and_fold(
        dataset_id,
        output_dir,
        model_name,
        postprocessor_name,
        fold,
        verbose
    )


def compute_metrics_from_preprocessed_data_model_postprocessor_all_folds(
        dataset_id: str,
        output_dir: str,
        model_name: str,
        postprocessor_name: str,
        verbose: bool = False
):
    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(
            process_fold,
            range(5),  # Folds 0 to 4
            [dataset_id] * 5,
            [output_dir] * 5,
            [model_name] * 5,
            [postprocessor_name] * 5,
            [verbose] * 5
        ))
    dataset_name = load_dataset_and_configuration(dataset_id)[0]
    summarize_metrics_from_model_and_postprocessor(dataset_name, model_name, postprocessor_name, output_dir)


def compute_metrics_from_model_name(
        dataset_id: str,
        output_dir: str,
        model_name: str,
        postprocessor_name: str = None,
        verbose: bool = False
):
    if postprocessor_name is not None:
        compute_metrics_from_preprocessed_data_model_postprocessor_all_folds(dataset_id, output_dir,
                                                                             model_name, postprocessor_name,
                                                                             verbose)
    else:
        postprocessor_names = ["ACLS", "CC"] if "SEMANTIC" in model_name else ["ConfLUNet"]

        for pp_name in postprocessor_names:
            compute_metrics_from_preprocessed_data_model_postprocessor_all_folds(dataset_id, output_dir, model_name, pp_name, verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation from model name and postprocessor name')
    parser.add_argument('--dataset_id', type=int, help='Dataset ID')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--postprocessor_name', type=str, default=None, help='Postprocessor name')
    parser.add_argument('--fold', type=int, default=None, help='Fold number')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.fold is None and args.postprocessor_name is None:
        compute_metrics_from_model_name(args.dataset_id, args.output_dir, args.model_name, None, args.verbose)
    elif args.fold is None:
        compute_metrics_from_preprocessed_data_model_postprocessor_all_folds(args.dataset_id, args.output_dir,
                                                                             args.model_name, args.postprocessor_name,
                                                                             args.verbose)
    else:
        raise NotImplementedError("Not implemented yet")
