import torch
from torch import nn
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class ConfLUNetLoss(nn.Module):
    def __init__(self, seg_loss, center_loss_weight: float, offsets_loss_weight: float, *args, **kwargs):
        super(ConfLUNetLoss, self).__init__(*args, **kwargs)
        self.seg_loss = seg_loss
        self.center_loss = nn.MSELoss()
        self.offsets_loss = nn.L1Loss(reduction='none')
        self.center_loss_weight = center_loss_weight
        self.offsets_loss_weight = offsets_loss_weight

    def forward(self, net_output, target):
        seg_output, center_output, offsets_output = net_output
        seg_target, center_target, offsets_target = target

        if type(seg_output) != type(seg_target):
            raise ValueError(f"Type mismatch between seg_output and seg_target: {type(seg_output)} != {type(seg_target)}"
                             f"Please ensure that the output and target are of the same type.")

        seg_loss = self.seg_loss(seg_output, seg_target)
        center_loss = self.center_loss(center_target, center_output)

        # Disregard voxels outside the GT segmentation
        positive_target = seg_target[:, 1, :].unsqueeze(1)
        offset_loss_weights_matrix = (positive_target.expand_as(offsets_target) == 1)
        offsets_loss = self.offsets_loss(offsets_target, offsets_output) * offset_loss_weights_matrix
        if offset_loss_weights_matrix.sum() > 0:
            offsets_loss = offsets_loss.sum() / offset_loss_weights_matrix.sum()
        else:  # No foreground voxels
            offsets_loss = offsets_loss.sum() * 0

        final_loss = (seg_loss
                      + self.center_loss_weight * center_loss
                      + self.offsets_loss_weight * offsets_loss)

        return final_loss



