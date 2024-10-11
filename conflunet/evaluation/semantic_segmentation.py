import argparse
import os
import pandas as pd
import numpy as np
import nibabel as nib
import warnings
from conflunet.evaluation.utils import find_confluent_lesions


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