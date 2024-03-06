import argparse
import os
import nibabel as nib
import pandas as pd
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from multiprocessing import Pool
from postprocess import remove_small_lesions_from_instance_segmentation
import warnings


def dice_metric(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Computes Dice coefficient for a single example.
    Args:
        ground_truth: numpy.ndarray, binary ground truth segmentation target, with shape [W, H, D].
        predictions: numpy.ndarray, binary segmentation predictions, with shape [W, H, D].
    Returns:
        float: Dice coefficient overlap in [0.0, 1.0] between ground_truth and predictions.
    """
    assert ground_truth.shape == predictions.shape, "Shapes of ground truth and predictions do not match."
    assert set(np.unique(predictions)).issubset({0, 1}), "predictions should be binary."
    assert set(np.unique(ground_truth)).issubset({0, 1}), "ground_truth should be binary."

    # Calculate intersection and union of y_true and y_predict
    intersection = np.sum(predictions * ground_truth)
    union = np.sum(predictions) + np.sum(ground_truth)

    # Calcualte dice metric
    if intersection == 0.0 and union == 0.0:
        dice = 1.0
    else:
        dice = (2. * intersection) / union

    return dice


def dice_norm_metric(ground_truth: np.ndarray, predictions: np.ndarray) -> tuple:
    """
    Compute Normalised Dice Coefficient (nDSC),
    False positive rate (FPR), False negative rate (FNR) for a single example.

    Args:
      ground_truth: numpy.ndarray, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions: numpy.ndarray, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      tuple: Normalised dice coefficient (float in [0.0, 1.0]),
             False positive rate (float in [0.0, 1.0]),
             False negative rate (float in [0.0, 1.0]),
             between ground_truth and predictions.
    """
    assert ground_truth.shape == predictions.shape, "Shapes of ground truth and predictions do not match."
    assert set(np.unique(predictions)).issubset({0, 1}), "predictions should be binary."
    assert set(np.unique(ground_truth)).issubset({0, 1}), "ground_truth should be binary."

    # Reference for normalized DSC
    r = 0.001
    # Cast to float32 type
    gt = ground_truth.astype("float32")
    seg = predictions.astype("float32")
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 1.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = k * fp
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn)
        return dsc_norm


def intersection_over_union(pred_mask: np.ndarray, ref_mask: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) for two masks.
    Args:
        pred_mask: numpy.ndarray, binary segmentation mask predicted by the model. Shape [H, W, D].
        ref_mask: numpy.ndarray, binary ground truth segmentation mask. Shape [H, W, D].
    Returns:
        float: Intersection over Union (IoU) score between the predicted mask and the reference mask.
    """
    assert pred_mask.shape == ref_mask.shape, "Shapes of pred_mask and ref_mask do not match."
    assert set(np.unique(pred_mask)).issubset({0,1}), "pred_mask should be binary."
    assert set(np.unique(ref_mask)).issubset({0,1}), "ref_mask should be binary."
    intersection = np.logical_and(pred_mask, ref_mask).sum()
    union = np.logical_or(pred_mask, ref_mask).sum()
    return intersection / union if union != 0 else 0


def match_instances(pred: np.ndarray, ref: np.ndarray, threshold: float = 0.1):
    """
    Match predicted instances to ground truth instances based on Intersection over Union (IoU) threshold.
    Args:
        pred: numpy.ndarray, instance segmentation mask of predicted instances. Shape [H, W, D].
        ref: numpy.ndarray, instance segmentation mask of ground truth instances. Shape [H, W, D].
        threshold: float, minimum IoU threshold for considering a match. Defaults to 0.1.
    Returns:
        Tuple: A tuple containing three lists:
               - matched_pairs: List of tuples (pred_id, ref_id) indicating matched instance pairs.
               - unmatched_pred: List of unmatched predicted instance ids.
               - unmatched_ref: List of unmatched ground truth instance ids.
    """
    assert pred.shape == ref.shape, "Shapes of pred and ref do not match."
    if not (0 <= threshold <= 1):
        warnings.warn(f"IoU threshold expected to be in the range [0, 1] but got {threshold}.", UserWarning)

    matched_pairs = []
    unmatched_pred = []
    unmatched_ref = []

    for pred_id in np.unique(pred):
        if pred_id == 0:  # skip background
            continue
        pred_mask = pred == pred_id

        max_iou = -np.inf
        matched_ref_id = None
        for ref_id in np.unique(ref):
            if ref_id == 0:  # skip background
                continue
            ref_mask = ref == ref_id
            iou = intersection_over_union(pred_mask, ref_mask)

            if iou > max_iou:
                max_iou = iou
                matched_ref_id = ref_id

        if max_iou > threshold:
            matched_pairs.append((pred_id, matched_ref_id))
        else:
            unmatched_pred.append(pred_id)

    for ref_id in np.unique(ref):
        if ref_id == 0 or ref_id in [x[1] for x in matched_pairs]:
            continue
        unmatched_ref.append(ref_id)

    return matched_pairs, unmatched_pred, unmatched_ref


def panoptic_quality(pred: np.ndarray, ref: np.ndarray, matched_pairs: list = None, unmatched_pred: list = None,
                     unmatched_ref: list = None):
    """
    Compute the Panoptic Quality (PQ) metric to compare predicted and reference instance segmentation.
    Args:
        pred: numpy.ndarray, instance segmentation mask of predicted instances. Shape [H, W, D].
        ref: numpy.ndarray, instance segmentation mask of ground truth instances. Shape [H, W, D].
        matched_pairs: list of tuples (pred_id, ref_id) indicating matched instance pairs.
                        If None, computes it. Defaults to None.
        unmatched_pred: list of unmatched predicted instance ids. If None, computes it. Defaults to None.
        unmatched_ref: list of unmatched ground truth instance ids. If None, computes it. Defaults to None.
    Returns:
        float: Panoptic Quality (PQ) metric.
    """
    assert pred.shape == ref.shape, "Shapes of pred and ref do not match."

    if matched_pairs is None or unmatched_pred is None or unmatched_ref is None:
        matched_pairs, unmatched_pred, unmatched_ref = match_instances(pred, ref)

    assert all([x[0] in np.unique(pred) and x[1] in np.unique(ref) for x in matched_pairs]), \
        "All instances in matched_pairs should be in pred and ref."
    assert all([x in np.unique(pred) for x in unmatched_pred]), "All instances in unmatched_pred should be in pred."
    assert all([x in np.unique(ref) for x in unmatched_ref]), "All instances in unmatched_ref should be in ref."

    tp = len(matched_pairs)
    fp = len(unmatched_pred)
    fn = len(unmatched_ref)

    sq_sum = 0
    for pair in matched_pairs:
        pred_mask = pred == pair[0]
        ref_mask = ref == pair[1]
        iou = intersection_over_union(pred_mask, ref_mask)
        sq_sum += iou

    sq = sq_sum / (tp + 1e-6)  # Segmentation Quality
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)  # Recognition Quality
    pq = sq * rq  # Panoptic Quality

    return pq


def f_beta_score(pred: np.ndarray = None, ref: np.ndarray = None, beta: float = 1,
                 matched_pairs: list = None, unmatched_pred: list = None, unmatched_ref: list = None):
    """
    Compute the F-beta score, a weighted harmonic mean of precision and recall.
    Args:
        pred: numpy.ndarray, instance segmentation mask of predicted instances. Shape [H, W, D]. Defaults to None.
        ref: numpy.ndarray, instance segmentation mask of ground truth instances. Shape [H, W, D]. Defaults to None.
        beta: float, weighting factor for precision in harmonic mean. Defaults to 1.
        matched_pairs: list of tuples (pred_id, ref_id) indicating matched instance pairs.
                        If None, computes it. Defaults to None.
        unmatched_pred: list of unmatched predicted instance ids. If None, computes it. Defaults to None.
        unmatched_ref: list of unmatched ground truth instance ids. If None, computes it. Defaults to None.
    Returns:
        float: F-beta score.
    """
    if matched_pairs is None or unmatched_pred is None or unmatched_ref is None:
        assert pred.shape == ref.shape, "Shapes of pred and ref do not match."
        matched_pairs, unmatched_pred, unmatched_ref = match_instances(pred, ref)

    assert all([x[0] in np.unique(pred) and x[1] in np.unique(ref) for x in matched_pairs]), \
        "All instances in matched_pairs should be in pred and ref."
    assert all([x in np.unique(pred) for x in unmatched_pred]), "All instances in unmatched_pred should be in pred."
    assert all([x in np.unique(ref) for x in unmatched_ref]), "All instances in unmatched_ref should be in ref."

    tp = len(matched_pairs)
    fp = len(unmatched_pred)
    fn = len(unmatched_ref)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    f_score = (1 + beta ** 2) * ((precision * recall) / ((beta ** 2 * precision) + recall + 1e-6))

    return f_score


def ltpr(pred: np.ndarray = None, ref: np.ndarray = None, matched_pairs: list = None,
         unmatched_ref: list = None):
    """
    Compute the Lesion True Positive Rate (LTPr), also known as recall but for object-wise metrics.
    Args:
        pred: numpy.ndarray, instance segmentation mask of predicted instances. Shape [H, W, D]. Defaults to None.
        ref: numpy.ndarray, instance segmentation mask of ground truth instances. Shape [H, W, D]. Defaults to None.
        matched_pairs: list of tuples (pred_id, ref_id) indicating matched instance pairs. Defaults to None.
        unmatched_ref: list of unmatched ground truth instance ids. Defaults to None.
    Returns:
        float: Lesion True Positive Rate (LTPR).
    """
    if matched_pairs is None or unmatched_ref is None:
        assert pred.shape == ref.shape, "Shapes of pred and ref do not match."
        matched_pairs, _, unmatched_ref = match_instances(pred, ref)

    assert all([x[0] in np.unique(pred) and x[1] in np.unique(ref) for x in matched_pairs]), \
        "All instances in matched_pairs should be in pred and ref."
    assert all([x in np.unique(ref) for x in unmatched_ref]), "All instances in unmatched_ref should be in ref."

    tp = len(matched_pairs)
    fn = len(unmatched_ref)
    return tp / (tp + fn + 1e-6)


def ppv(pred: np.ndarray = None, ref: np.ndarray = None, matched_pairs: list = None, unmatched_pred: list = None):
    """
    Compute the Positive Predictive Value (PPV), also known as precision but for object-wise metrics.
    Args:
        pred: numpy.ndarray, instance segmentation mask of predicted instances. Shape [H, W, D]. Defaults to None.
        ref: numpy.ndarray, instance segmentation mask of ground truth instances. Shape [H, W, D]. Defaults to None.
        matched_pairs: list of tuples (pred_id, ref_id) indicating matched instance pairs. Defaults to None.
        unmatched_pred: list of unmatched predicted instance ids. Defaults to None.
    Returns:
        float: Positive Predictive Value (PPV).
    """
    if matched_pairs is None and unmatched_pred is None:
        assert pred.shape == ref.shape, "Shapes of pred and ref do not match."
        matched_pairs, unmatched_pred, _ = match_instances(pred, ref)

    assert all([x[0] in np.unique(pred) and x[1] in np.unique(ref) for x in matched_pairs]), \
        "All instances in matched_pairs should be in pred and ref."
    assert all([x in np.unique(pred) for x in unmatched_pred]), "All instances in unmatched_pred should be in pred."

    tp = len(matched_pairs)
    fp = len(unmatched_pred)
    return tp / (tp + fp + 1e-6)


def dice_per_tp(pred: np.ndarray, ref: np.ndarray, matched_pairs: list):
    """
    Compute Dice score for each matched lesion.
    Args:
        pred: numpy.ndarray, instance segmentation mask of predicted instances. Shape [H, W, D].
        ref: numpy.ndarray, instance segmentation mask of ground truth instances. Shape [H, W, D].
        matched_pairs: list of tuples (pred_id, ref_id) indicating matched instance pairs.
    Returns:
        list: Dice scores for each matched lesion.
    """
    assert pred.shape == ref.shape, "Shapes of pred and ref do not match."
    assert all([x[0] in np.unique(pred) and x[1] in np.unique(ref) for x in matched_pairs]), \
        "All instances in matched_pairs should be in pred and ref."
    dice_scores = []

    for pair in matched_pairs:
        pred_mask = pred == pair[0]
        ref_mask = ref == pair[1]
        score = dice_metric(pred_mask, ref_mask)
        dice_scores.append(score)

    return dice_scores


def pred_lesion_count(pred: np.ndarray):
    """
    Retrieves the predicted lesion count.
    Args:
        pred: numpy.ndarray, instance segmentation mask of predicted instances. Shape [H, W, D].
    Returns:
        int: The count of predicted lesions.
    """
    # The unique values represent different lesions. 0 typically represents the background.
    unique_lesions = len(set(pred.flatten())) - (1 if 0 in pred else 0)
    return unique_lesions


def ref_lesion_count(ref: np.ndarray):
    """
    Retrieves the reference lesion count.
    Args:
        ref: numpy.ndarray, instance segmentation mask of reference instances. Shape [H, W, D].
    Returns:
        int: The count of reference lesions.
    """
    # The unique values represent different lesions. 0 typically represents the background.
    unique_lesions = len(set(ref.flatten())) - (1 if 0 in ref else 0)
    return unique_lesions


def DiC(pred: np.ndarray, ref: np.ndarray):
    """
    Computes the absolute difference in lesion counting between prediction and reference.
    Args:
        pred: numpy.ndarray, instance segmentation mask of predicted instances. Shape [H, W, D].
        ref: numpy.ndarray, instance segmentation mask of reference instances. Shape [H, W, D].
    Returns:
        int: The absolute difference in lesion count.
    """
    assert pred.shape == ref.shape, "Shapes of pred and ref do not match."

    pred_count = pred_lesion_count(pred)
    ref_count = ref_lesion_count(ref)
    return abs(pred_count - ref_count)


def find_confluent_lesions(instance_segmentation: np.ndarray) -> list:
    """
    Find confluent lesions ids in the instance segmentation.
    Args:
        instance_segmentation: numpy.ndarray, instance segmentation with shape [H, W, D].
    Returns:
        list: List of ids of the confluent connected components.
    """
    # Remove background 0s
    instances = np.unique(instance_segmentation)[1:]
    num_instances = len(instances)

    binary_segmentation = np.copy(instance_segmentation)
    binary_segmentation[binary_segmentation > 0] = 1

    connected_components, num_connected_components = label(binary_segmentation)
    connected_component_ids = np.unique(connected_components)[1:]  # Remove background 0s

    if num_connected_components == num_instances:
        # No confluent lesions found
        return []

    # Use multiprocessing to speed up the process
    pool = Pool()
    results = []

    for instance_id in instances:
        results.append(pool.apply_async(check_confluent, (instance_id, connected_component_ids,
                                                          instance_segmentation, connected_components)))

    pool.close()
    pool.join()

    # Retrieve the results from multiprocessing
    confluent_connected_component_ids = []
    for result in results:
        confluent_connected_component_ids.append(result.get())

    confluent_connected_component_ids = [instance_id for instance_id, cc_ids in confluent_connected_component_ids if
                                         len(cc_ids) > 0]

    return confluent_connected_component_ids


def check_confluent(instance_id: int, connected_component_ids: list,
                    instance_segmentation: np.ndarray, connected_components: np.ndarray) -> tuple:
    """
    Check if a lesion instance is part of a larger confluent lesion.
    Args:
        instance_id: int, id of the lesion instance.
        connected_component_ids: list, list of connected components ids.
        instance_segmentation: numpy.ndarray, instance segmentation with shape [H, W, D].
        connected_components: numpy.ndarray, mask of connected components with shape [H, W, D]
            where each id corresponds to a connected component.
    Returns:
        tuple: Tuple containing the instance id and a list of ids of confluent connected components.
    """
    confluent_connected_component_ids = []
    for cc_id in connected_component_ids:
        instance_indices = np.where(instance_segmentation == instance_id)
        cc_indices = np.where(connected_components == cc_id)

        instance_coords = set(zip(instance_indices[0], instance_indices[1], instance_indices[2]))
        cc_coords = set(zip(cc_indices[0], cc_indices[1], cc_indices[2]))

        # Check if the instance is completely in the connected component
        if instance_coords.issubset(cc_coords):
            confluent = True
            if all([len(instance_indices[i]) == len(cc_indices[i]) for i in range(len(instance_indices))]):
                if all(np.array_equal(instance_indices[i], cc_indices[i]) for i in range(len(instance_indices))):
                    confluent = False

            if confluent:
                confluent_connected_component_ids.append(cc_id)

    return instance_id, confluent_connected_component_ids


def compute_metrics(args):
    # Check if prediction and reference folders exist
    if not os.path.exists(args.pred_path) or not os.path.exists(args.ref_path):
        print("Either prediction or reference path doesn't exist!")
        return

    metrics_dict = {"Subject_ID": [], "File": []}
    if args.dsc or args.all: metrics_dict["DSC"] = []
    if args.pq or args.all: metrics_dict["PQ"] = []
    if args.fbeta or args.all: metrics_dict["Fbeta"] = []
    if args.ltpr or args.all: metrics_dict["LTPR"] = []
    if args.ppv or args.all: metrics_dict["PPV"] = []
    if args.dice_per_tp or args.all: metrics_dict["Dice_Per_TP"] = []
    if args.pred_count or args.all: metrics_dict["Pred_Lesion_Count"] = []
    if args.ref_count or args.all: metrics_dict["Ref_Lesion_Count"] = []
    if args.dic or args.all: metrics_dict["DiC"] = []
    if args.clr or args.all: metrics_dict["CLR"] = []
    if args.dice_per_tp_cl or args.all: metrics_dict["Dice_Per_TP_CL"] = []
    # if args.clm or args.all: metrics_dict["CLM"] = []
    all_pred_matches = {"Subject_ID": [], "Lesion_ID": [], "Ref_Lesion_ID_Match": [], "Volume_Pred": [],
                        "Volume_Ref": [], "DSC": []}
    all_ref_matches = {"Subject_ID": [], "Lesion_ID": [], "Pred_Lesion_ID_Match": [], "Volume_Ref": [],
                       "Volume_Pred": [], "DSC": []}
    metrics_dict["CL_Count"] = []

    dd = "test" if args.test else "val"
    ref_dir = os.path.join(args.ref_path, dd, "labels")

    for ref_file in sorted(os.listdir(ref_dir)):
        if ref_file.endswith("mask-instances.nii.gz"):
            print(ref_file)
            subj_id = ref_file.split("_ses")[0].split("sub-")[-1]  # Extracting subject ID
            pred_file = "sub-" + subj_id + "_ses-01_pred_instances.nii.gz"
            pred_file_path = os.path.join(args.pred_path, pred_file)

            if not os.path.exists(pred_file_path):
                pred_file_path = pred_file_path.replace('pred_instances', 'pred-instances')
            if not os.path.exists(pred_file_path):
                print(f"No prediction found for {pred_file}")
                continue

            ref_img = nib.load(os.path.join(ref_dir, ref_file))
            voxel_size = ref_img.header.get_zooms()
            ref_img = remove_small_lesions_from_instance_segmentation(ref_img.get_fdata(), voxel_size,
                                                                      l_min=args.minimum_lesion_size)
            pred_img = nib.load(pred_file_path).get_fdata()

            matched_pairs, unmatched_pred, unmatched_ref = match_instances(pred_img, ref_img)

            metrics_dict["Subject_ID"].append(subj_id)
            metrics_dict["File"].append(pred_file)
            if args.dsc or args.all:
                dsc = dice_metric((ref_img > 0).astype(np.uint8), (pred_img > 0).astype(np.uint8))
                metrics_dict["DSC"].append(dsc)
            if args.pq or args.all:
                pq_val = panoptic_quality(pred=pred_img, ref=ref_img,
                                          matched_pairs=matched_pairs, unmatched_pred=unmatched_pred,
                                          unmatched_ref=unmatched_ref)
                metrics_dict["PQ"].append(pq_val)
            if args.fbeta or args.all:
                fbeta_val = f_beta_score(matched_pairs=matched_pairs, unmatched_pred=unmatched_pred,
                                         unmatched_ref=unmatched_ref)
                metrics_dict["Fbeta"].append(fbeta_val)
            if args.ltpr or args.all:
                ltpr_val = ltpr(matched_pairs=matched_pairs, unmatched_ref=unmatched_ref)
                metrics_dict["LTPR"].append(ltpr_val)
            if args.ppv or args.all:
                ppv_val = ppv(matched_pairs=matched_pairs, unmatched_pred=unmatched_pred)
                metrics_dict["PPV"].append(ppv_val)
            if args.dice_per_tp or args.all:
                dice_scores = dice_per_tp(pred_img, ref_img, matched_pairs)
                # Assuming you want the average Dice score per subject
                avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
                metrics_dict["Dice_Per_TP"].append(avg_dice)
            if args.pred_count or args.all:
                metrics_dict["Pred_Lesion_Count"].append(pred_lesion_count(pred_img))
            if args.ref_count or args.all:
                metrics_dict["Ref_Lesion_Count"].append(ref_lesion_count(ref_img))
            if args.dic or args.all:
                metrics_dict["DiC"].append(DiC(pred_img, ref_img))
            if args.clr or args.dice_per_tp_cl or args.all:
                confluents_ref_img = np.copy(ref_img)
                cl_ids = find_confluent_lesions(confluents_ref_img)

                # set all other ids to 0 in ref_img
                for id in np.unique(confluents_ref_img):
                    if id not in cl_ids:
                        confluents_ref_img[confluents_ref_img == id] = 0

                matched_pairs_cl, unmatched_pred_cl, unmatched_ref_cl = match_instances(pred_img, confluents_ref_img)

                clm = len(cl_ids)
                metrics_dict["CL_Count"].append(clm)
                if args.clr or args.all:
                    if clm == 0:
                        metrics_dict["CLR"].append(np.nan)
                    else:
                        clr = ltpr(matched_pairs=matched_pairs_cl, unmatched_ref=unmatched_ref_cl)
                        metrics_dict["CLR"].append(clr)
                if args.dice_per_tp_cl or args.all:
                    if clm == 0:
                        metrics_dict["Dice_Per_TP_CL"].append(np.nan)
                    else:
                        dice_scores_cl = dice_per_tp(pred_img, confluents_ref_img, matched_pairs_cl)
                        # Assuming you want the average Dice score per subject
                        avg_dice_cl = sum(dice_scores_cl) / len(dice_scores_cl) if dice_scores_cl else 0
                        metrics_dict["Dice_Per_TP_CL"].append(avg_dice_cl)

            # Store for every predicted lesion the potential match in the reference annotation,
            # along with both lesion volumes
            all_pred_labels, all_pred_counts = [m[1:] for m in np.unique(pred_img, return_counts=True)]
            pred_to_ref_matches = dict(matched_pairs)

            # If it was not already computed, compute the dice scores for each TP predicted lesion
            if not 'dice_scores' in locals():
                dice_scores = dice_per_tp(pred_img, ref_img, matched_pairs)

            for pid, volume_pred in zip(all_pred_labels, all_pred_counts):
                all_pred_matches["Subject_ID"].append(subj_id)
                all_pred_matches["Lesion_ID"].append(pid)

                if not (pid in unmatched_pred):
                    matched_ref_id = pred_to_ref_matches[pid]
                    volume_ref = sum((ref_img[ref_img == matched_ref_id] > 0).astype(np.uint8))
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
            all_ref_labels, all_ref_counts = [m[1:] for m in np.unique(ref_img, return_counts=True)]
            ref_to_pred_matches = {v: k for k, v in dict(matched_pairs).items()}
            for rid, volume_ref in zip(all_ref_labels, all_ref_counts):
                all_ref_matches["Subject_ID"].append(subj_id)
                all_ref_matches["Lesion_ID"].append(rid)

                if not (rid in unmatched_ref):
                    matched_pred_id = ref_to_pred_matches[rid]
                    volume_pred = sum((pred_img[pred_img == matched_pred_id] > 0).astype(np.uint8))
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

    model_name = os.path.basename(os.path.dirname(args.pred_path))
    # Convert dictionary to dataframe and save as CSV
    df = pd.DataFrame(metrics_dict)
    df.to_csv(os.path.join(args.pred_path, f"metrics_{model_name}_{dd}.csv"), index=False)

    df = pd.DataFrame(all_pred_matches)
    df.to_csv(os.path.join(args.pred_path, f"predicted_lesions_matches_{model_name}_{dd}.csv"), index=False)

    df = pd.DataFrame(all_ref_matches)
    df.to_csv(os.path.join(args.pred_path, f"reference_lesions_matches_{model_name}_{dd}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute segmentation metrics.")
    parser.add_argument("--pred_path", required=True, help="Path to the directory with prediction files.")
    parser.add_argument("--ref_path", required=True,
                        help="Path to the directory with reference files (containing val/ and test/).")
    parser.add_argument("--minimum_lesion_size", default=14,
                        help="Minimum lesion size.")
    parser.add_argument("--test", action="store_true", help="Wether to use the test data or not. Default is val data.")
    parser.add_argument("--dsc", action="store_true", help="Compute Dice Score (DSC).")
    parser.add_argument("--pq", action="store_true", help="Compute Panoptic Quality (PQ).")
    parser.add_argument("--fbeta", action="store_true", help="Compute F-beta score.")
    parser.add_argument("--ltpr", action="store_true", help="Compute Lesion True Positive Rate (LTPR).")
    parser.add_argument("--ppv", action="store_true", help="Compute Positive Predictive Value (PPV).")
    parser.add_argument("--dice_per_tp", action="store_true", help="Compute Dice score for each true positive lesion.")
    parser.add_argument("--pred_count", action="store_true", help="Retrieve the predicted lesion count.")
    parser.add_argument("--ref_count", action="store_true", help="Retrieve the reference lesion count.")
    parser.add_argument("--dic", action="store_true", help="Compute the absolute difference in lesion counting.")
    parser.add_argument("--clr", action="store_true", help="Compute confluent lesion recall.")
    parser.add_argument("--dice_per_tp_cl", action="store_true",
                        help="Compute Dice score for each true positive confluent lesion.")
    parser.add_argument("--clm", action="store_true",
                        help="Compute CLM (Placeholder for next big metric, currently number of confluent lesions).")
    parser.add_argument("--all", action="store_true", help="Compute all available metrics.")

    args = parser.parse_args()
    compute_metrics(args)