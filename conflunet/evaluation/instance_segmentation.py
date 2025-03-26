import numpy as np
from typing import List

from conflunet.evaluation.utils import match_instances, intersection_over_union
from conflunet.evaluation.semantic_segmentation import dice_metric


def dice_per_tp(pred: np.ndarray, ref: np.ndarray, matched_pairs: list) -> List[float]:
    """
    Compute Dice score for each matched lesion.
    Args:
        pred: numpy.ndarray, instance segmentation mask of predicted instances. Shape [H, W, D].
        ref: numpy.ndarray, instance segmentation mask of ground truth instances. Shape [H, W, D].
        matched_pairs: list of tuples (pred_id, ref_id) indicating matched instance pairs.
    Returns:
        list: Dice scores for each matched lesion.
    """
    assert pred.shape == ref.shape, \
        "Shapes of pred and ref do not match ({} != {}).".format(pred.shape, ref.shape)
    assert all([x[0] in np.unique(pred) and x[1] in np.unique(ref) for x in matched_pairs]), \
        "All instances in matched_pairs should be in pred and ref."
    dice_scores = []

    for pair in matched_pairs:
        pred_mask = pred == pair[0]
        ref_mask = ref == pair[1]
        score = dice_metric(pred_mask, ref_mask)
        dice_scores.append(score)

    return dice_scores


def panoptic_quality(pred: np.ndarray, ref: np.ndarray, matched_pairs: list = None, unmatched_pred: list = None,
                     unmatched_ref: list = None) -> float:
    """
    Compute the Panoptic Quality (PQ) metric to compare predicted and reference instance segmentation.
    In this version of the function, we are not considering the background.
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
    assert pred.shape == ref.shape, \
        "Shapes of pred and ref do not match ({} != {}).".format(pred.shape, ref.shape)

    # Check if pred and ref contain no objects
    if np.max(pred) == 0 and np.max(ref) == 0:
        return 1.0  # Convention: If no objects exist in both, PQ is perfect (1.0).

    if matched_pairs is None or unmatched_pred is None or unmatched_ref is None:
        matched_pairs, unmatched_pred, unmatched_ref = match_instances(pred, ref)

    unique_preds = np.unique(pred)
    unique_refs = np.unique(ref)

    if pred is not None and ref is not None:
        for pred_id, ref_id in matched_pairs:
            assert pred_id in unique_preds, f"ID {pred_id} is not in prediction matrix"
            assert ref_id in unique_refs, f"ID {ref_id} is not in reference matrix"
        assert all([x in unique_preds for x in unmatched_pred]), "All instances in unmatched_pred should be in pred."
        assert all([x in unique_refs for x in unmatched_ref]), "All instances in unmatched_ref should be in ref."

    tp = len(matched_pairs)
    fp = len(unmatched_pred)
    fn = len(unmatched_ref)

    # Edge case: No GT instances but predictions exist → PQ = 0
    if len(unique_refs) == 1 and unique_refs[0] == 0 and fp > 0:
        return 0.0

    # Edge case: No predictions but GT instances exist → PQ = 0
    if len(unique_preds) == 1 and unique_preds[0] == 0 and fn > 0:
        return 0.0

    sq_sum = 0
    for pair in matched_pairs:
        pred_mask = pred == pair[0]
        ref_mask = ref == pair[1]
        iou = intersection_over_union(pred_mask, ref_mask)
        sq_sum += iou

    sq = sq_sum / max(tp, 1e-6)  # Segmentation Quality
    rq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-6)  # Recognition Quality
    pq = sq * rq  # Panoptic Quality

    return pq
