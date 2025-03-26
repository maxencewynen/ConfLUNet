import numpy as np

from conflunet.evaluation.utils import match_instances


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
        assert pred.shape == ref.shape, \
            "Shapes of pred and ref do not match ({} != {}).".format(pred.shape, ref.shape)
        matched_pairs, unmatched_pred, unmatched_ref = match_instances(pred, ref)

    if pred is not None and ref is not None:
        unique_preds = np.unique(pred)
        unique_refs = np.unique(ref)
        for pred_id, ref_id in matched_pairs:
            assert pred_id in unique_preds, f"ID {pred_id} is not in prediction matrix"
            assert ref_id in unique_refs, f"ID {ref_id} is not in reference matrix"
        assert all([x in unique_preds for x in unmatched_pred]), "All instances in unmatched_pred should be in pred."
        assert all([x in unique_refs for x in unmatched_ref]), "All instances in unmatched_ref should be in ref."

    tp = len(matched_pairs)
    fp = len(unmatched_pred)
    fn = len(unmatched_ref)

    _precision = tp / (tp + fp + 1e-6)
    _recall = tp / (tp + fn + 1e-6)

    f_score = (1 + beta ** 2) * ((_precision * _recall) / ((beta ** 2 * _precision) + _recall + 1e-6))

    return f_score


def recall(pred: np.ndarray = None, ref: np.ndarray = None, matched_pairs: list = None,
           unmatched_ref: list = None):
    """
    Compute the Lesion True Positive Rate (recall), also known as recall but for object-wise metrics.
    Args:
        pred: numpy.ndarray, instance segmentation mask of predicted instances. Shape [H, W, D]. Defaults to None.
        ref: numpy.ndarray, instance segmentation mask of ground truth instances. Shape [H, W, D]. Defaults to None.
        matched_pairs: list of tuples (pred_id, ref_id) indicating matched instance pairs. Defaults to None.
        unmatched_ref: list of unmatched ground truth instance ids. Defaults to None.
    Returns:
        float: Lesion True Positive Rate (recall).
    """
    if matched_pairs is None or unmatched_ref is None:
        assert pred.shape == ref.shape, \
            "Shapes of pred and ref do not match ({} != {}).".format(pred.shape, ref.shape)
        matched_pairs, _, unmatched_ref = match_instances(pred, ref)

    if pred is not None and ref is not None:
        unique_preds = np.unique(pred)
        unique_refs = np.unique(ref)
        for pred_id, ref_id in matched_pairs:
            assert pred_id in unique_preds, f"ID {pred_id} is not in prediction matrix"
            assert ref_id in unique_refs, f"ID {ref_id} is not in reference matrix"
        assert all([x in unique_refs for x in unmatched_ref]), "All instances in unmatched_ref should be in ref."

    tp = len(matched_pairs)
    fn = len(unmatched_ref)
    if tp == 0 and fn == 0:
        return 1.0
    return tp / (tp + fn + 1e-6)


def precision(pred: np.ndarray = None, ref: np.ndarray = None, matched_pairs: list = None, unmatched_pred: list = None):
    """
    Compute the Positive Predictive Value (precision), also known as precision but for object-wise metrics.
    Args:
        pred: numpy.ndarray, instance segmentation mask of predicted instances. Shape [H, W, D]. Defaults to None.
        ref: numpy.ndarray, instance segmentation mask of ground truth instances. Shape [H, W, D]. Defaults to None.
        matched_pairs: list of tuples (pred_id, ref_id) indicating matched instance pairs. Defaults to None.
        unmatched_pred: list of unmatched predicted instance ids. Defaults to None.
    Returns:
        float: Positive Predictive Value (precision).
    """
    if matched_pairs is None and unmatched_pred is None:
        assert pred.shape == ref.shape, \
            "Shapes of pred and ref do not match ({} != {}).".format(pred.shape, ref.shape)
        matched_pairs, unmatched_pred, _ = match_instances(pred, ref)

    if pred is not None and ref is not None:
        unique_preds = np.unique(pred)
        unique_refs = np.unique(ref)
        for pred_id, ref_id in matched_pairs:
            assert pred_id in unique_preds, f"ID {pred_id} is not in prediction matrix"
            assert ref_id in unique_refs, f"ID {ref_id} is not in reference matrix"
        assert all([x in unique_preds for x in unmatched_pred]), "All instances in unmatched_pred should be in pred."

    tp = len(matched_pairs)
    fp = len(unmatched_pred)
    if tp == 0 and fp == 0:
        return 1.0
    return tp / (tp + fp + 1e-6)


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
    assert pred.shape == ref.shape, \
        "Shapes of pred and ref do not match ({} != {}).".format(pred.shape, ref.shape)

    pred_count = pred_lesion_count(pred)
    ref_count = ref_lesion_count(ref)
    return abs(pred_count - ref_count)