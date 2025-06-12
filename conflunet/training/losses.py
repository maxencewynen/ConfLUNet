import torch
import warnings
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from typing import Callable
from monai.losses import DiceLoss


class WeightedSemanticSegmentationLoss(Callable):
    def __init__(
            self,
            dice_loss_weight: float = 0.5,
            focal_loss_weight: float = 1.0,
            gamma: float = 2.0
    ):
        super(WeightedSemanticSegmentationLoss, self).__init__()
        self.loss_function_dice = DiceLoss(to_onehot_y=True,
                                           softmax=True, sigmoid=False,
                                           include_background=False)
        self.ce_loss = CrossEntropyLoss(reduction='none')
        self.dice_loss_weight = dice_loss_weight
        self.focal_loss_weight = focal_loss_weight
        self.gamma = gamma

    def __call__(
            self,
            prediction: torch.Tensor,
            reference: torch.Tensor,
            weights: torch.Tensor = None
    ):
        # Dice loss
        dice_loss = self.loss_function_dice(prediction, reference)
        
        # Focal loss
        ce = self.ce_loss(prediction, torch.squeeze(reference, dim=1))
        pt = torch.exp(-ce)
        loss2 = (1 - pt) ** self.gamma * ce
        if weights is not None:
            loss2 *= torch.squeeze(weights, dim=1)
        focal_loss = torch.mean(loss2)

        # Combine losses
        segmentation_loss = self.dice_loss_weight * dice_loss + self.focal_loss_weight * focal_loss
        return segmentation_loss, dice_loss, focal_loss


class SemanticSegmentationLoss(WeightedSemanticSegmentationLoss):
    def __init__(
            self,
            dice_loss_weight: float = 0.5,
            focal_loss_weight: float = 1.0,
            gamma: float = 2.0
    ):
        super(SemanticSegmentationLoss, self).__init__(dice_loss_weight=dice_loss_weight,
                                                       focal_loss_weight=focal_loss_weight,
                                                       gamma=gamma)

    def __call__(
            self,
            prediction: torch.Tensor,
            reference: torch.Tensor,
            weights: torch.Tensor = None
    ):
        return super(SemanticSegmentationLoss, self).__call__(prediction, reference, weights=None)


class WeightedConfLUNetLoss(Callable):
    def __init__(
            self,
            segmentation_loss_weight: float = 1.0,
            offsets_loss_weight: float = 1.0,
            center_heatmap_loss_weight: float = 1.0,
            loss_function_segmentation: Callable = WeightedSemanticSegmentationLoss(),
            loss_function_offsets: Callable = L1Loss(reduction='none'),
            loss_function_center_heatmap: Callable = MSELoss(reduction='none'),
    ):
        super(WeightedConfLUNetLoss, self).__init__()
        self.segmentation_loss_weight = segmentation_loss_weight
        self.offsets_loss_weight = offsets_loss_weight
        self.center_heatmap_loss_weight = center_heatmap_loss_weight
        self.loss_function_segmentation = loss_function_segmentation
        self.loss_function_offsets = loss_function_offsets
        self.loss_function_center_heatmap = loss_function_center_heatmap

    def __call__(
            self,
            semantic_pred: torch.Tensor,
            center_heatmap_pred: torch.Tensor,
            offsets_pred: torch.Tensor,
            semantic_ref: torch.Tensor,
            center_heatmap_ref: torch.Tensor,
            offsets_ref: torch.Tensor,
            semantic_weights: torch.Tensor = None,
            offsets_weights: torch.Tensor = None,
            centers_weights: torch.Tensor = None,
            offsets_roi: torch.Tensor = None # to discard offsets in areas where the instance segmentation is not present (unsplittable lesions)
    ):
        if offsets_roi is None:
            offsets_roi = semantic_ref
        ### SEGMENTATION LOSS ###
        segmentation_loss, dice_loss, focal_loss = self.loss_function_segmentation(semantic_pred, semantic_ref, semantic_weights)

        ### CENTER OF MASS LOSS ###
        center_heatmap_loss = self.loss_function_center_heatmap(center_heatmap_pred, center_heatmap_ref)
        if centers_weights is not None:
            center_heatmap_loss *= centers_weights
        # Disregard voxels inside semantic_ref but outside offsets_roi
        only_included_voxels = torch.logical_not(torch.logical_xor(semantic_ref, offsets_roi))
        center_heatmap_loss *= only_included_voxels
        center_heatmap_loss = torch.mean(center_heatmap_loss)

        ### OFFSETS LOSS ###
        # Disregard voxels outside the GT segmentation
        offset_loss_weights_matrix = offsets_roi.expand_as(offsets_pred)
        offset_loss = self.loss_function_offsets(offsets_pred, offsets_ref) * offset_loss_weights_matrix

        if offsets_weights is not None:
            offset_loss *= offsets_weights
        if offset_loss_weights_matrix.sum() > 0:
            offset_loss = offset_loss.sum() / offset_loss_weights_matrix.sum()
        else:  # No foreground voxels
            offset_loss = offset_loss.sum() * 0

        ### TOTAL LOSS ###
        loss = (self.segmentation_loss_weight * segmentation_loss +
                (self.center_heatmap_loss_weight * center_heatmap_loss) +
                (self.offsets_loss_weight * offset_loss))

        return loss, dice_loss, focal_loss, segmentation_loss, center_heatmap_loss, offset_loss


class ConfLUNetLoss(WeightedConfLUNetLoss):
    def __init__(
            self,
            segmentation_loss_weight: float = 1.0,
            offsets_loss_weight: float = 1.0,
            center_heatmap_loss_weight: float = 1.0,
            loss_function_segmentation: Callable = WeightedSemanticSegmentationLoss(),
            loss_function_offsets: Callable = L1Loss(reduction='none'),
            loss_function_center_heatmap: Callable = MSELoss(reduction='none'),
    ):
        super(WeightedConfLUNetLoss, self).__init__()
        self.segmentation_loss_weight = segmentation_loss_weight
        self.offsets_loss_weight = offsets_loss_weight
        self.center_heatmap_loss_weight = center_heatmap_loss_weight
        self.loss_function_segmentation = loss_function_segmentation
        self.loss_function_offsets = loss_function_offsets
        self.loss_function_center_heatmap = loss_function_center_heatmap

    def __call__(
            self,
            semantic_pred: torch.Tensor,
            center_heatmap_pred: torch.Tensor,
            offsets_pred: torch.Tensor,
            semantic_ref: torch.Tensor,
            center_heatmap_ref: torch.Tensor,
            offsets_ref: torch.Tensor,
            semantic_weights: torch.Tensor = None,
            offsets_weights: torch.Tensor = None,
            centers_weights: torch.Tensor = None,
            offsets_roi: torch.Tensor = None # to discard offsets in areas where the instance segmentation is not present (unsplittable lesions)
    ):
        if semantic_weights is not None:
            warnings.warn("ConfLUNetLoss does not support semantic_weights. Ignoring them.")
        if offsets_weights is not None:
            warnings.warn("ConfLUNetLoss does not support offsets_weights. Ignoring them.")
        if centers_weights is not None:
            warnings.warn("ConfLUNetLoss does not support centers_weights. Ignoring them.")
        return super(ConfLUNetLoss, self).__call__(semantic_pred, center_heatmap_pred, offsets_pred,
                                                   semantic_ref, center_heatmap_ref, offsets_ref,
                                                   semantic_weights=None, offsets_weights=None, 
                                                   centers_weights=None)
