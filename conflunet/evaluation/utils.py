import json
import numpy as np
import warnings
import pandas as pd
from os.path import join as pjoin
from scipy.ndimage import label, binary_dilation
from typing import List, Tuple, Dict
from conflunet.postprocessing.utils import convert_types

METRICS_TO_AVERAGE = ["PQ", "DSC", "nDSC", "F1", "Recall", "Precision", "FPR", "Dice_Per_TP", "DiC", "F1_CLU",
                      "Recall_CLU", "Precision_CLU", "Dice_Per_TP_CLU", "PQ_CLU"]
METRICS_TO_AVERAGE += [f"{metric}_tier_1" for metric in METRICS_TO_AVERAGE if "CLU" in metric]
METRICS_TO_AVERAGE += [f"{metric}_tier_2" for metric in METRICS_TO_AVERAGE if "CLU" in metric and "tier_1" not in metric]

METRICS_TO_SUM = ["TP", "FP", "FN", "Pred_Lesion_Count", "Ref_Lesion_Count", "CLU_Count", "TP_CLU", "FN_CLU"]
METRICS_TO_SUM += [f"{metric}_tier_1" for metric in METRICS_TO_SUM if "CLU" in metric]
METRICS_TO_SUM += [f"{metric}_tier_2" for metric in METRICS_TO_SUM if "CLU" in metric and "tier_1" not in metric]
METRICS_TO_SUM += ["FP_CLU"]


def intersection_over_union(pred_mask: np.ndarray, ref_mask: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) for two masks.
    Args:
        pred_mask: numpy.ndarray, binary segmentation mask predicted by the model. Shape [H, W, D].
        ref_mask: numpy.ndarray, binary ground truth segmentation mask. Shape [H, W, D].
    Returns:
        float: Intersection over Union (IoU) score between the predicted mask and the reference mask.
    """
    assert pred_mask.shape == ref_mask.shape, f"Shapes of pred_mask and ref_mask do not match. ({pred_mask.shape} != {ref_mask.shape})"
    # assert set(np.unique(pred_mask)).issubset({0,1}), "pred_mask should be binary."
    # assert set(np.unique(ref_mask)).issubset({0,1}), "ref_mask should be binary."
    intersection = np.logical_and(pred_mask, ref_mask).sum()
    union = np.logical_or(pred_mask, ref_mask).sum()
    return intersection / union if union != 0 else 0


def find_tierx_confluent_instances(instance_segmentation: np.ndarray, tier: int = 1):
    """
    Find Tier X confluent instances, defined as instances that are part of a confluent lesion when dilated X times.
    """
    binary_segmentation = np.copy(instance_segmentation)
    binary_segmentation[binary_segmentation > 0] = 1
    binary_segmentation = binary_segmentation.astype(np.bool_)
    if tier == 0:
        binary_segmentation_dilated = binary_segmentation
    else:
        binary_segmentation_dilated = binary_dilation(binary_segmentation, iterations=tier)

    connected_components_dilated, num_connected_components_dilated = label(binary_segmentation_dilated)

    confluent_instances = []
    for cc_id in range(1, num_connected_components_dilated + 1):
        this_cc = connected_components_dilated == cc_id

        this_cc = instance_segmentation[this_cc]
        unique_values = np.unique(this_cc)
        unique_values = list(unique_values[unique_values != 0])

        if len(unique_values) > 1:
            confluent_instances += unique_values

    return sorted(list(set(confluent_instances)))


def find_confluent_instances(instance_segmentation: np.ndarray):
    """
    Find confluent instances in an instance segmentation map. 30x faster than the previous implementation.
    :param instance_segmentation:
    :return: list of instance ids that are a unit of a confluent lesion
    """
    return find_tierx_confluent_instances(instance_segmentation, tier=0)


def find_confluent_lesions(instance_segmentation):
    return find_confluent_instances(instance_segmentation)


def filter_matched_pairs(
        matched_pairs: List[Tuple[int, int, float]]
) -> (List[Tuple[int, int, float]], List[Tuple[int, int, float]]):
    best_tuples = {}
    removed_tuples = []

    # Iterate over data to find the best tuples for each reference_id
    for tup in matched_pairs:
        predicted_id, reference_id, iou = tup
        # If reference_id is not in best_tuples or current IoU is higher, update
        if reference_id not in best_tuples or iou > best_tuples[reference_id][2]:
            if reference_id in best_tuples:
                removed_tuples.append(best_tuples[reference_id])  # Add previous best to removed
            best_tuples[reference_id] = tup
        else:
            removed_tuples.append(tup)  # Add lower IoU tuple directly to removed

    # Final lists
    filtered_tuples = list(best_tuples.values())
    return filtered_tuples, removed_tuples


def match_instances(
        pred: np.ndarray,
        ref: np.ndarray,
        threshold: float = 0.1,
        return_iou: bool = False,
        return_removed_matched_pred: bool = False
) -> (List[Tuple[int, int]], List[int], List[int]):
    """
    Match predicted instances to reference instances based on Intersection over Union (IoU) threshold.
    -> If multiple predicted instances are matched to the same reference instance, the pair with the highest IoU is kept
    and other instances are considered as wrong/FP (added in unmatched_pred).
    -> If multiple reference instances are matched to the same predicted instance, the pair with the highest IoU is kept
    and the other instances are considered as missed/FN (added in unmatched_ref).
    Args:
        pred: numpy.ndarray, instance segmentation mask of predicted instances. Shape [H, W, D].
        ref: numpy.ndarray, instance segmentation mask of reference instances. Shape [H, W, D].
        threshold: float, minimum IoU threshold for considering a match. Defaults to 0.1.
        return_iou: bool, whether to return the IoU values for matched pairs. Defaults to False.
    Returns:
        Tuple: A tuple containing three lists:
               - matched_pairs: List of tuples (pred_id, ref_id) indicating matched instance pairs. If return_iou is True,
                                 the tuple will be (pred_id, ref_id, iou).
               - removed_matched_pred: List of tuples (pred_id, ref_id) indicating matched instance pairs that were
                                             removed due to lower IoU with another reference instance.
               - unmatched_pred: List of all falsely predicted instance ids (FP). (includes pred instances from
                                             removed_matched_pred)
               - unmatched_ref: List of unmatched reference instance ids (FN).
    """
    assert pred.shape == ref.shape, f"Shapes of pred and ref do not match ({pred.shape} != {ref.shape})."
    if not (0 <= threshold <= 1):
        warnings.warn(f"IoU threshold expected to be in the range [0, 1] but got {threshold}.", UserWarning)

    matched_pairs = []
    unmatched_pred = []
    unmatched_ref = []
    unique_refs = np.unique(ref)

    for pred_id in np.unique(pred):
        if pred_id == 0:  # skip background
            continue
        pred_mask = pred == pred_id

        unique_refs_to_check = np.unique(ref[pred_mask])

        max_iou = -np.inf
        matched_ref_id = None
        for ref_id in unique_refs_to_check:
            if ref_id == 0:  # skip background
                continue
            ref_mask = ref == ref_id
            iou = intersection_over_union(pred_mask, ref_mask)

            if iou > max_iou:
                max_iou = iou
                matched_ref_id = ref_id
            if max_iou == 1:
                break  # perfect match found

        if max_iou > threshold:
            matched_pairs.append((pred_id, matched_ref_id, max_iou))
        else:
            unmatched_pred.append(pred_id)

    matched_pairs, removed_pairs = filter_matched_pairs(matched_pairs)
    for removed_pair in removed_pairs:
        if removed_pair[0] not in unmatched_pred:
            unmatched_pred.append(removed_pair[0])

    if not return_iou:
        matched_pairs = [(x[0], x[1]) for x in matched_pairs]
        removed_matched_pred = [(x[0], x[1]) for x in removed_pairs]
    else:
        removed_matched_pred = removed_pairs

    for ref_id in unique_refs:
        if ref_id == 0 or ref_id in [x[1] for x in matched_pairs]:
            continue
        unmatched_ref.append(ref_id)

    if return_removed_matched_pred:
        return matched_pairs, removed_matched_pred, unmatched_pred, unmatched_ref

    return matched_pairs, unmatched_pred, unmatched_ref


def summarize_metrics_from_model_and_postprocessor(dataset_name, model_name, postprocessor_name, save_dir):
    from conflunet.postprocessing.utils import convert_types

    all_metrics = {}
    all_pred_matches = []
    all_ref_matches = []
    for fold in range(5):
        fold_metrics_file = pjoin(save_dir, dataset_name, model_name, f"fold_{fold}", postprocessor_name,
                                  "metrics_summary.json")
        with open(fold_metrics_file, 'r') as f:
            this_fold_metrics = json.load(f)
            all_metrics[f"fold_{fold}"] = this_fold_metrics

        fold_pred_matches_file = pjoin(save_dir, dataset_name, model_name, f"fold_{fold}", postprocessor_name,
                                       "pred_matches.csv")
        df_pred_matches = pd.read_csv(fold_pred_matches_file)
        df_pred_matches["fold"] = fold
        all_pred_matches.append(df_pred_matches)

        fold_ref_matches_file = pjoin(save_dir, dataset_name, model_name, f"fold_{fold}", postprocessor_name,
                                      "ref_matches.csv")
        df_ref_matches = pd.read_csv(fold_ref_matches_file)
        df_ref_matches["fold"] = fold
        all_ref_matches.append(df_ref_matches)

    metrics_folds_details_file = pjoin(save_dir, dataset_name, model_name, f"metrics_fold_details_{postprocessor_name}.json")
    with open(metrics_folds_details_file, 'w') as f:
        json.dump(convert_types(all_metrics), f, indent=4)

    # compute mean metrics or sum when applicable
    metrics_summary = {}
    metrics_summary.update(
        {metric: np.mean([d[metric] for d in all_metrics.values()]) for metric in METRICS_TO_AVERAGE})
    metrics_summary.update({metric: np.sum([d[metric] for d in all_metrics.values()]) for metric in METRICS_TO_SUM})

    metrics_summary_file = pjoin(save_dir, dataset_name, model_name, f"metrics_avg_across_folds_{postprocessor_name}.json")
    with open(metrics_summary_file, 'w') as f:
        json.dump(convert_types(metrics_summary), f, indent=4)

    metrics_std = {}
    metrics_std.update({metric: np.std([d[metric] for d in all_metrics.values()]) for metric in METRICS_TO_AVERAGE})
    metrics_std_file = pjoin(save_dir, dataset_name, model_name, f"metrics_std_across_folds_{postprocessor_name}.json")
    with open(metrics_std_file, 'w') as f:
        json.dump(convert_types(metrics_std), f, indent=4)

    df_pred_matches = pd.concat(all_pred_matches)
    df_pred_matches.to_csv(pjoin(save_dir, dataset_name, model_name, f"pred_lesions_infos_{postprocessor_name}.csv"), index=False)

    df_ref_matches = pd.concat(all_ref_matches)
    df_ref_matches.to_csv(pjoin(save_dir, dataset_name, model_name, f"ref_lesions_infos_{postprocessor_name}.csv"), index=False)


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
    # also save as csv
    df_pred_matches = pd.DataFrame([
        {"patient_id": patient_id, "pred_lesion_id": lesion_id,
         **{key: patient_data[key][i] for key in patient_data if key != "Lesion_ID"}}
        for patient_id, patient_data in all_pred_matches.items()
        for i, lesion_id in enumerate(patient_data["Lesion_ID"])
    ])
    df_pred_matches["TP"] = ~df_pred_matches["Ref_Lesion_ID_Match"].isna()
    df_pred_matches["FP"] = df_pred_matches["Ref_Lesion_ID_Match"].isna()
    df_pred_matches.to_csv(pjoin(save_dir, "pred_matches.csv"), index=False)

    ref_matches_file = pjoin(save_dir, "ref_matches.json")
    with open(ref_matches_file, 'w') as f:
        json.dump(convert_types(all_ref_matches), f, indent=4)
    # also save as csv
    df_ref_matches = pd.DataFrame([
        {"patient_id": patient_id, "ref_lesion_id": lesion_id,
         **{key: patient_data[key][i] for key in patient_data if key != "Lesion_ID"}}
        for patient_id, patient_data in all_ref_matches.items()
        for i, lesion_id in enumerate(patient_data["Lesion_ID"])
    ])
    df_ref_matches["FN"] = df_ref_matches["Pred_Lesion_ID_Match"].isna()
    df_ref_matches.to_csv(pjoin(save_dir, "ref_matches.csv"), index=False)

    # compute mean metrics or sum when applicable
    metrics_summary = {}
    metrics_summary.update({metric: np.nanmean([d[metric] for d in all_metrics.values()]) for metric in METRICS_TO_AVERAGE})
    metrics_summary.update({metric: np.nansum([d[metric] for d in all_metrics.values()]) for metric in METRICS_TO_SUM})

    metrics_summary_file = pjoin(save_dir, "metrics_summary.json")
    with open(metrics_summary_file, 'w') as f:
        json.dump(convert_types(metrics_summary), f, indent=4)

