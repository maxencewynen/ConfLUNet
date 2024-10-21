import numpy as np


def dice_metric(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Computes Dice coefficient for a single example.
    Args:
        ground_truth: numpy.ndarray, binary ground truth segmentation target, with shape [W, H, D].
        predictions: numpy.ndarray, binary segmentation predictions, with shape [W, H, D].
    Returns:
        float: Dice coefficient overlap in [0.0, 1.0] between ground_truth and predictions.
    """
    assert ground_truth.shape == predictions.shape, \
        "Shapes of ground truth and predictions do not match ({} != {}).".format(ground_truth.shape, predictions.shape)
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


def dice_norm_metric(predictions: np.ndarray, ground_truth: np.ndarray) -> tuple:
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
    assert ground_truth.shape == predictions.shape, \
        "Shapes of ground truth and predictions do not match ({} != {}).".format(ground_truth.shape, predictions.shape)
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
