import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from typing import Callable
from monai.losses import DiceLoss


class SemanticSegmentationLoss(Callable):
    def __init__(
            self,
            dice_loss_weight: float = 0.5,
            focal_loss_weight: float = 1.0,
            gamma: float = 2.0
    ):
        super(SemanticSegmentationLoss, self).__init__()
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
            reference: torch.Tensor
    ):
        # Dice loss
        dice_loss = self.loss_function_dice(prediction, reference)

        # Focal loss
        ce = self.ce_loss(prediction, torch.squeeze(reference, dim=1))
        pt = torch.exp(-ce)
        loss2 = (1 - pt) ** self.gamma * ce
        focal_loss = torch.mean(loss2)

        # Combine losses
        segmentation_loss = self.dice_loss_weight * dice_loss + self.focal_loss_weight * focal_loss
        return segmentation_loss, dice_loss, focal_loss


class ConfLUNetLoss(Callable):
    def __init__(
            self,
            segmentation_loss_weight: float = 1.0,
            offsets_loss_weight: float = 1.0,
            center_heatmap_loss_weight: float = 1.0,
            loss_function_segmentation: Callable = SemanticSegmentationLoss(),
            loss_function_offsets: Callable = L1Loss(reduction='none'),
            loss_function_center_heatmap: Callable = MSELoss(),
    ):
        super(ConfLUNetLoss, self).__init__()
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
            offsets_ref: torch.Tensor
    ):
        ### SEGMENTATION LOSS ###
        segmentation_loss, dice_loss, focal_loss = self.loss_function_segmentation(semantic_pred, semantic_ref)

        ### CENTER OF MASS LOSS ###
        center_heatmap_loss = self.loss_function_center_heatmap(center_heatmap_pred, center_heatmap_ref)

        ### OFFSETS LOSS ###
        # Disregard voxels outside the GT segmentation
        offset_loss_weights_matrix = semantic_ref.expand_as(offsets_pred)
        offset_loss = self.loss_function_offsets(offsets_pred, offsets_ref) * offset_loss_weights_matrix

        if offset_loss_weights_matrix.sum() > 0:
            offset_loss = offset_loss.sum() / offset_loss_weights_matrix.sum()
        else:  # No foreground voxels
            offset_loss = offset_loss.sum() * 0

        ### TOTAL LOSS ###
        loss = (self.segmentation_loss_weight * segmentation_loss +
                (self.center_heatmap_loss_weight * center_heatmap_loss) +
                (self.offsets_loss_weight * offset_loss))

        return loss, dice_loss, focal_loss, segmentation_loss, center_heatmap_loss, offset_loss
