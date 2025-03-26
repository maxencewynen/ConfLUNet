import os
import argparse
import warnings
import numpy as np
import nibabel as nib
from pprint import pprint
from os.path import join as pjoin

from conflunet.evaluation.metrics import compute_metrics
from conflunet.evaluation.utils import save_metrics
from conflunet.postprocessing.small_instances_removal import remove_small_lesions_from_instance_segmentation


def evaluate_single_prediction(pred_file, ref_file, verbose=False):
    instance_seg_pred = nib.load(pred_file)
    voxel_size_pred = instance_seg_pred.header["pixdim"][1:4]
    instance_seg_pred = instance_seg_pred.get_fdata()

    instance_seg_ref = nib.load(ref_file)
    voxel_size_ref = instance_seg_ref.header["pixdim"][1:4]
    instance_seg_ref = instance_seg_ref.get_fdata()

    if not np.allclose(voxel_size_ref, voxel_size_pred):
        warnings.warn(f"Reference annotation file {ref_file} and predicted annotation file {pred_file} "
                      f"have different voxel sizes. This might be an issue worth investigating.")

    instance_seg_ref = remove_small_lesions_from_instance_segmentation(instance_seg_ref, voxel_size=voxel_size_ref)

    return compute_metrics(
        instance_seg_pred,
        instance_seg_ref,
        voxel_size=voxel_size_ref,
        verbose=verbose
    )


def find_matching_prediction_file(predictions_dir, ref_file):
    # Expecting ref_file name to be {case_identifier}.nii.gz
    if not ref_file.endswith(".nii.gz"):
        extension = '.'.join(ref_file.split('.')[1:])
        raise ValueError(f"extension not recognized. expected .nii.gz but got .{extension}")

    case_identifier = os.path.basename(ref_file).replace('.nii.gz', '')
    matching_pred_file = []
    for filename in os.listdir(predictions_dir):
        if case_identifier not in filename:
            continue
        matching_pred_file.append(filename)

    if len(matching_pred_file) == 0:
        raise FileNotFoundError(f"Cannot find matching prediction file for reference annotation {ref_file}")

    if len(matching_pred_file) > 1:
        raise ValueError(f"Found multiple matching prediction files for the same reference annotation {case_identifier}"
                         f": ({matching_pred_file})")

    return matching_pred_file[0]


def main(args):
    all_metrics = {}
    all_pred_matches = {}
    all_ref_matches = {}

    if os.path.isfile(args.pred):
        if not os.path.isfile(args.ref):
            raise ValueError(f"`pred` argument is a file while `ref` argument is a directory ({args.pred}, {args.ref})")

        case_identifier = os.path.basename(args.ref).replace('.nii.gz', '')
        metrics, pred_matches, ref_matches = evaluate_single_prediction(args.pred, args.ref, verbose=args.verbose)
        save_dir = os.path.dirname(args.pred)

        all_metrics[case_identifier] = metrics
        all_pred_matches[case_identifier] = pred_matches
        all_ref_matches[case_identifier] = ref_matches

    else:
        predictions_dir = args.pred
        reference_dir = args.ref
        if not os.path.isdir(args.ref):
            raise ValueError(f"`pred` argument is a directory while `ref` argument is a file ({args.pred}, {args.ref})")

        for ref_file in sorted(os.listdir(reference_dir)):
            print(ref_file)
            if os.path.isdir(pjoin(reference_dir, ref_file)) or not ref_file.endswith('.nii.gz'):
                continue
            pred_file = find_matching_prediction_file(predictions_dir, ref_file)
            pred_file = pjoin(predictions_dir, pred_file)
            ref_file = pjoin(reference_dir, ref_file)
            case_identifier = os.path.basename(ref_file).replace('.nii.gz', '')

            metrics, pred_matches, ref_matches = evaluate_single_prediction(pred_file, ref_file, verbose=args.verbose)

            all_metrics[case_identifier] = metrics
            all_pred_matches[case_identifier] = pred_matches
            all_ref_matches[case_identifier] = ref_matches

        save_dir = predictions_dir

    pprint(all_metrics)
    save_metrics(all_metrics, all_pred_matches, all_ref_matches, save_dir=save_dir)
    print(f"Saved metrics files in {save_dir}")


def evaluate_entry_point():
    parser = argparse.ArgumentParser(
        description='Compute metrics between reference instance annotation and predictions')
    parser.add_argument('--ref', type=str, help='Reference instance annotations directory/file')
    parser.add_argument('--pred', type=str, help='Predicted instance masks directory/file')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    evaluate_entry_point()

