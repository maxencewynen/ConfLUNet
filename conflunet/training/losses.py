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
    ):
        breakpoint()
        ### SEGMENTATION LOSS ###
        segmentation_loss, dice_loss, focal_loss = self.loss_function_segmentation(semantic_pred, semantic_ref, semantic_weights)

        ### CENTER OF MASS LOSS ###
        center_heatmap_loss = self.loss_function_center_heatmap(center_heatmap_pred, center_heatmap_ref)
        if centers_weights is not None:
            center_heatmap_loss *= centers_weights
        center_heatmap_loss = torch.mean(center_heatmap_loss)

        ### OFFSETS LOSS ###
        # Disregard voxels outside the GT segmentation
        offset_loss_weights_matrix = semantic_ref.expand_as(offsets_pred)
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


class PullLoss(Callable):
    def __call__(
            self,
            offsets_pred: torch.Tensor,
            associative_pred: torch.Tensor,
            semantic_ref: torch.Tensor,
            instance_seg_ref: torch.Tensor,
    ):
        ### BUILD 4D EMBEDDING ###
        # concatenate (position matrix + offsets) and associative pred
        complete_embedding = ...  # (batch_size, 4, H, W, D)

        # Useful for later: build a 4d instance reference mask where the ids are repeated along the associative
        # dimension (3 spatial + 1 associative)
        instance_seg_ref_plus_associative = ...

        total_pull_loss = 0

        all_instances_id = torch.unique(instance_seg_ref)
        all_instances_id = all_instances_id[all_instances_id != 0]
        n_instances = len(all_instances_id)

        # For each instance in the reference:
        for instance_id in all_instances_id:
            # Have a binary mask for this instance
            this_instance_mask = instance_seg_ref_plus_associative == instance_id

            # Compute centroid coordinate (4d tensor), which corresponds to the average of all embedding coordinates
            # of the instances voxels
            this_instance_centroid_coordinate = ...

            # Get list of 4d embedding coordinates for this instance
            this_instance_embeddings_coordinates = ...  # (n_voxels_in_this_instance, 4)

            # This step can be done in multiple chunks if GPU memory is needed!
            # Average all squared L2 norm of (embedding coordinate - this instance centroid coordinate) for all voxels
            this_instance_pull_loss = MSELoss()(this_instance_embeddings_coordinates,
                                                this_instance_centroid_coordinate.expand_as(this_instance_embeddings_coordinates))

        total_pull_loss = total_pull_loss / n_instances if n_instances > 0 else torch.tensor(0).to(offsets_pred.device)

        return total_pull_loss


class PushLoss(Callable):
    def __init__(self, margin: float = 4):
        super(PushLoss, self).__init__()
        self.margin = margin

    def __call__(
            self,
            offsets_pred: torch.Tensor,
            associative_pred: torch.Tensor,
            semantic_ref: torch.Tensor,
            instance_seg_ref: torch.Tensor,
    ):
        # Compute spatial embedding
        spatial_embedding = ...  # position matrix + offsets pred

        total_push_loss = 0

        all_instances_id = torch.unique(instance_seg_ref)
        all_instances_id = all_instances_id[all_instances_id != 0]
        n_instances = len(all_instances_id)

        spatial_centroids = {}
        associative_centroids = {}
        # compute the spatial and associative centroids of each instance
        for instance_id in all_instances_id:
            # Have a binary mask for this instance
            this_instance_mask_spatial = instance_seg_ref == instance_id


            spatial_centroids[instance_id] = ...

            this_instance_mask_associative = instance_seg_ref == instance_id
            associative_centroids[instance_id] = ...

        for instance_id_1 in all_instances_id:
            spatial_centroid_1 = spatial_centroids[instance_id_1]
            associative_centroid_1 = associative_centroids[instance_id_1]

            for instance_id_2 in all_instances_id[all_instances_id != instance_id_1]:
                spatial_centroid_2 = spatial_centroids[instance_id_2]
                associative_centroid_2 = associative_centroids[instance_id_2]

                # L2 norm between spatial centroids
                spatial_distance = torch.norm(spatial_centroid_1 - spatial_centroid_2)

                # compute the right margin for this pair of instances
                associative_margin = torch.sqrt(self.margin ** 2 - spatial_distance ** 2)  # ?????

                associative_distance = torch.norm(associative_centroid_1 - associative_centroid_2)
                push_loss = torch.relu(torch.abs(associative_margin - associative_distance) ** 2)

                total_push_loss += push_loss

        total_push_loss = total_push_loss / (n_instances * (n_instances - 1)) if n_instances > 0 \
            else torch.tensor(0).to(offsets_pred.device)

        return total_push_loss


class HybridLoss(Callable):
    def __init__(
            self,
            segmentation_loss_weight: float = 1.0,
            embedding_loss_weight: float = 0.01, # lambda_emb
            loss_function_pull: Callable = PullLoss(),
            loss_function_push: Callable = PushLoss(),
            loss_function_segmentation: Callable = WeightedSemanticSegmentationLoss(),
            margin: float = 4,
    ):
        self.segmentation_loss_weight = segmentation_loss_weight
        self.embedding_loss_weight = embedding_loss_weight
        self.loss_function_segmentation = loss_function_segmentation
        self.loss_function_pull = loss_function_pull
        self.loss_function_push = loss_function_push
        self.margin = margin

    def __call__(
            self,
            semantic_pred: torch.Tensor,
            associative_pred: torch.Tensor,
            offsets_pred: torch.Tensor,
            semantic_ref: torch.Tensor,
            spatial_centers_ref: torch.Tensor,
            instance_seg_ref: torch.Tensor,
            semantic_weights: torch.Tensor = None,
            offsets_weights: torch.Tensor = None,
            centers_weights: torch.Tensor = None,
    ):
        segmentation_loss, dice_loss, focal_loss = self.loss_function_segmentation(semantic_pred, semantic_ref, semantic_weights)

        loss_pull = self.loss_function_pull(offsets_pred, associative_pred, semantic_ref, instance_seg_ref)
        loss_push = self.loss_function_push(offsets_pred, associative_pred, semantic_ref, instance_seg_ref)
        loss_embedding = loss_pull + loss_push

        loss = (self.segmentation_loss_weight * segmentation_loss +
                self.embedding_loss_weight * loss_embedding)

        return loss, dice_loss, focal_loss, segmentation_loss, loss_pull, loss_push, loss_embedding
