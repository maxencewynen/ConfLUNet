import argparse
import time
import numpy as np
from typing import Tuple, Dict, List

from conflunet.evaluation.utils import match_instances
from conflunet.evaluation.utils import find_confluent_lesions
from conflunet.evaluation.semantic_segmentation import dice_metric, dice_norm_metric
from conflunet.evaluation.instance_segmentation import panoptic_quality, dice_per_tp
from conflunet.evaluation.detection import f_beta_score, recall, precision, pred_lesion_count, ref_lesion_count, DiC


def vprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def compute_metrics(
        instance_pred: np.array,
        instance_ref: np.array,
        voxel_size: tuple = (1, 1, 1),
        verbose: bool = False
) -> Tuple[Dict[str, float], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Compute metrics for semantic, instance and detection segmentation tasks. The following metrics are computed:
    - Dice Similarity Coefficient (DSC)
    - Normalized Dice Similarity Coefficient (nDSC)
    - Panoptic Quality (PQ)
    - F1 score
    - Recall
    - Precision
    - Dice per true positive (Dice_Per_TP)
    - Predicted Lesion Count (Pred_Lesion_Count)
    - Reference Lesion Count (Ref_Lesion_Count)
    - Detection in Confluent Lesion Units (DiC)
    - Confluent Lesion Units Count (CLU_Count)
    - Recall in Confluent Lesion Units (Recall_CLU)
    - Precision in Confluent Lesion Units (Precision_CLU)
    - Dice per true positive in Confluent Lesion Units (Dice_Per_TP_CLU)
    - Per lesion match and volume information

    Also returns all_pred_matches and all_ref_matches, which contain information about the match between predicted and
    reference lesions, including the volume of each lesion and the Dice Similarity Coefficient (DSC) between them.
    Args:
        instance_pred:
        instance_ref:
        voxel_size:
        verbose:

    Returns:

    """
    ###########################################
    ###########      Metrics     ##############
    ###########################################
    metrics = {}
    vprint(verbose, f"[INFO] Matching instances...")
    matched_pairs, unmatched_pred, unmatched_ref = match_instances(instance_pred, instance_ref)
    vprint(verbose, f"[INFO] Found {len(matched_pairs)} matched pairs, {len(unmatched_pred)} unmatched predicted instances, "
                    f"and {len(unmatched_ref)} unmatched reference instances.")

    ### Compute metrics ###
    vprint(verbose, f"[INFO] Computing semantic segmentation metrics...")
    # Dice score
    dsc = dice_metric((instance_pred > 0).astype(np.uint8), (instance_ref > 0).astype(np.uint8))
    metrics["DSC"] = dsc
    
    # normalized Dice score
    ndsc = dice_norm_metric((instance_pred > 0).astype(np.uint8), (instance_ref > 0).astype(np.uint8))
    metrics["nDSC"] = ndsc
    start = time.time()
    dice_scores = dice_per_tp(instance_pred, instance_ref, matched_pairs)
    avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0 # average per patient
    metrics["Dice_Per_TP"] = avg_dice
    vprint(verbose, f"[INFO] Dice per TP computation took {time.time() - start:.2f} seconds.")

    vprint(verbose, f"[INFO] Computing instance segmentation metrics...")
    # PQ
    pq_val = panoptic_quality(pred=instance_pred, ref=instance_ref,
                              matched_pairs=matched_pairs, unmatched_pred=unmatched_pred,
                              unmatched_ref=unmatched_ref)
    metrics["PQ"] = pq_val

    vprint(verbose, f"[INFO] Computing detection segmentation metrics...")
    # F1 score
    f1 = f_beta_score(matched_pairs=matched_pairs, unmatched_pred=unmatched_pred, unmatched_ref=unmatched_ref)
    metrics["F1"] = f1

    # Recall and Precision
    recall_val = recall(matched_pairs=matched_pairs, unmatched_ref=unmatched_ref)
    metrics["Recall"] = recall_val
    precision_val = precision(matched_pairs=matched_pairs, unmatched_pred=unmatched_pred)
    metrics["Precision"] = precision_val

    # Additional metrics

    metrics["Pred_Lesion_Count"] = pred_lesion_count(instance_pred)
    metrics["Ref_Lesion_Count"] = ref_lesion_count(instance_ref)
    metrics["DiC"] = DiC(instance_pred, instance_ref)

    vprint(verbose, f"[INFO] Computing confluent lesion units metrics...")
    ### Confluent lesions Metrics ###
    confluents_instance_ref = np.copy(instance_ref)
    cl_ids = find_confluent_lesions(confluents_instance_ref)

    # set all other ids to 0 in instance_ref
    for id in np.unique(confluents_instance_ref):
        if id not in cl_ids:
            confluents_instance_ref[confluents_instance_ref == id] = 0

    matched_pairs_cl, unmatched_pred_cl, unmatched_ref_cl = match_instances(instance_pred, confluents_instance_ref)

    clm = len(cl_ids)
    metrics["CLU_Count"] = clm
    if clm == 0:
        metrics["Recall_CLU"] = np.nan
        metrics["Precision_CLU"] = np.nan
        metrics["Dice_Per_TP_CLU"] = np.nan
    else:
        clr = recall(matched_pairs=matched_pairs_cl, unmatched_ref=unmatched_ref_cl)
        clp = precision(matched_pairs=matched_pairs_cl, unmatched_pred=unmatched_pred_cl)
        metrics['TP_CLU'] = len(matched_pairs_cl)
        metrics["Recall_CLU"] = clr
        metrics["Precision_CLU"] = clp

        dice_scores_cl = dice_per_tp(instance_pred, confluents_instance_ref, matched_pairs_cl)
        avg_dice_cl = sum(dice_scores_cl) / len(dice_scores_cl) if dice_scores_cl else 0
        metrics["Dice_Per_TP_CLU"] = avg_dice_cl

    ###########################################
    ## Per lesion match & volume information ##
    ###########################################
    all_pred_matches = {"Lesion_ID": [], "Ref_Lesion_ID_Match": [], "Volume_Pred": [], "Volume_Ref": [], "DSC": []}
    all_ref_matches = {"Lesion_ID": [], "Pred_Lesion_ID_Match": [], "Volume_Ref": [], "Volume_Pred": [], "DSC": []}

    # Store for every predicted lesion the potential match in the reference annotation,
    # along with both lesion volumes
    all_pred_labels, all_pred_counts = np.unique(instance_pred[instance_pred != 0], return_counts=True)
    pred_to_ref_matches = dict(matched_pairs)

    for pid, volume_pred in zip(all_pred_labels, all_pred_counts):
        all_pred_matches["Lesion_ID"].append(pid)

        if not (pid in unmatched_pred):
            matched_ref_id = pred_to_ref_matches[pid]
            volume_ref = sum((instance_ref[instance_ref == matched_ref_id] > 0).astype(np.uint8))
            volume_ref *= np.prod(voxel_size)
            this_pairs_dsc = dice_scores[matched_pairs.index((pid, matched_ref_id))]
        else:
            matched_ref_id = None
            volume_ref = None
            this_pairs_dsc = None

        all_pred_matches["Ref_Lesion_ID_Match"].append(matched_ref_id)
        all_pred_matches["Volume_Pred"].append(volume_pred * np.prod(voxel_size))
        all_pred_matches["Volume_Ref"].append(volume_ref)
        all_pred_matches["DSC"].append(this_pairs_dsc)

    # Store for every lesion in the reference annotation the potential match in the predicted instance map,
    # along with both lesion volumes
    all_ref_labels, all_ref_counts = np.unique(instance_ref[instance_ref != 0], return_counts=True)
    ref_to_pred_matches = {v: k for k, v in dict(matched_pairs).items()}
    for rid, volume_ref in zip(all_ref_labels, all_ref_counts):
        all_ref_matches["Lesion_ID"].append(rid)

        if not (rid in unmatched_ref):
            matched_pred_id = ref_to_pred_matches[rid]
            volume_pred = sum((instance_pred[instance_pred == matched_pred_id] > 0).astype(np.uint8))
            volume_pred *= np.prod(voxel_size)
            this_pairs_dsc = dice_scores[matched_pairs.index((matched_pred_id, rid))]
        else:
            matched_pred_id = None
            volume_pred = None
            this_pairs_dsc = None

        all_ref_matches["Pred_Lesion_ID_Match"].append(matched_pred_id)
        all_ref_matches["Volume_Ref"].append(volume_ref * np.prod(voxel_size))
        all_ref_matches["Volume_Pred"].append(volume_pred)
        all_ref_matches["DSC"].append(this_pairs_dsc)

    return metrics, all_pred_matches, all_ref_matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics.")
    parser.add_argument("--pred_path", required=True, help="Path to the predicted instance segmentation mask (.nii.gz).")
    parser.add_argument("--ref_path", required=True, help="Path to the reference instance segmentation mask (.nii.gz).")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")

    args = parser.parse_args()
    import nibabel as nib
    pred = nib.load(args.pred_path)
    voxel_size = pred.header.get_zooms()
    pred = pred.get_fdata()
    ref = nib.load(args.ref_path).get_fdata()
    metrics, all_pred_matches, all_ref_matches = compute_metrics(pred, ref, voxel_size=voxel_size, verbose=args.verbose)
    print(f"{metrics=}", end="\n\n")
    print(f"{all_pred_matches=}", end="\n\n")
    print(f"{all_ref_matches=}")
