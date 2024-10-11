import numpy as np
from scipy.ndimage import label
import warnings
from conflunet.evaluation.semantic_segmentation import intersection_over_union


def find_confluent_instances(instance_segmentation: np.ndarray):
    """
    Find confluent instances in an instance segmentation map. 30x faster than the previous implementation.
    :param instance_segmentation:
    :return: list of instance ids that are a unit of a confluent lesion
    """
    binary_segmentation = np.copy(instance_segmentation)
    binary_segmentation[binary_segmentation > 0] = 1
    binary_segmentation = binary_segmentation.astype(np.bool_)

    connected_components, num_connected_components = label(binary_segmentation)

    confluent_instances = []
    for cc_id in range(1, num_connected_components + 1):
        this_cc = connected_components == cc_id

        this_cc = instance_segmentation[this_cc]
        unique_values = np.unique(this_cc)
        unique_values = list(unique_values[unique_values != 0])

        if len(unique_values) > 1:
            confluent_instances += unique_values

    return sorted(list(set(confluent_instances)))


def find_confluent_lesions(instance_segmentation):
    return find_confluent_instances(instance_segmentation)


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

