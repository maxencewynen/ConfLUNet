import torch
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import label
from typing import Dict, Union, Tuple

from monai.config.type_definitions import NdarrayOrTensor

from conflunet.postprocessing.basic_postprocessor import Postprocessor


class ConfLUNetPostprocessor(Postprocessor):
    def __init__(
            self,
            minimum_instance_size: int = 0,
            minimum_size_along_axis: int = 0,
            voxel_spacing: Tuple[float, float, float] = (1, 1, 1),
            semantic_threshold: float = 0.5,
            heatmap_threshold: float = 0.1,
            nms_kernel_size: int = 3,
            top_k: int = None,
            compute_voting: bool = False,
            calibrate_offsets: bool = False,
            device: torch.device = None,
            verbose: bool = True
    ):
        """
        ConfLUNet postprocessor
            heatmap threshold: A Float, threshold applied to center heatmap score.
            nms_kernel: An Integer, NMS max pooling kernel size.
            top_k: An Integer, top k centers to keep. If None, all centers > threshold are kept
        """
        super(ConfLUNetPostprocessor, self).__init__(
            minimum_instance_size=minimum_instance_size,
            minimum_size_along_axis=minimum_size_along_axis,
            voxel_spacing=voxel_spacing,
            semantic_threshold=semantic_threshold,
            name="ConfLUNet",
            device=device,
            verbose=verbose
        )
        assert 0 <= heatmap_threshold <= 1, "Threshold should be between 0 and 1"
        assert top_k is None or top_k > 0, "top_k should be None or a positive integer"
        assert (nms_kernel_size % 2) == 1, "NMS kernel must be odd"
        
        self.heatmap_threshold = heatmap_threshold
        self.nms_kernel_size = nms_kernel_size
        self.top_k = top_k
        self.compute_voting = compute_voting
        self.calibrate_offsets = calibrate_offsets

    def _find_instance_center(self, ctr_hmp: torch.Tensor) -> torch.Tensor:
        """
        Inspired from https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py
        Find the center points from the center heatmap.
        Arguments:
            ctr_hmp: A Tensor of shape [N, 1, H, W, D] of raw center heatmap output, where N is the batch size,
                for consistent, we only support N=1.

        Returns:
            A Tensor of shape [K, 3] where K is the number of center points. The order of second dim is (x, y, z).
        """
        if ctr_hmp.size(0) != 1:
            raise ValueError('Only supports inference for batch size = 1')

        # thresholding, setting values below threshold to -1
        ctr_hmp = F.threshold(ctr_hmp, self.heatmap_threshold, -1)

        # NMS
        nms_padding = (self.nms_kernel_size - 1) // 2
        ctr_hmp_max_pooled = F.max_pool3d(ctr_hmp, kernel_size=self.nms_kernel_size, stride=1, padding=nms_padding)
        ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1

        # squeeze first two dimensions
        ctr_hmp = ctr_hmp.squeeze()
        assert len(ctr_hmp.size()) == 3, f'Expected ctr_hmp to have 3 dimensions but got {len(ctr_hmp.size())}'

        # find non-zero elements
        nonzeros = (ctr_hmp > 0).short()

        # Find clusters of centers to consider them as one center instead of two
        centers_labeled, num_centers = label(nonzeros.cpu().numpy())
        centers_labeled = torch.from_numpy(centers_labeled).to(nonzeros.device)
        for c in list(range(1, num_centers + 1)):
            coords_cx, coords_cy, coords_cz = torch.where(centers_labeled == c)

            # if center is made of two voxels or more
            if len(coords_cx) > 1:
                # keep only one center voxel at random, since all of them have the same probability
                # of being a center
                coord_to_keep = np.random.choice(list(range(len(coords_cx))))

                # set all the other center voxels to zero
                for i in range(len(coords_cx)):
                    if i != coord_to_keep:
                        ctr_hmp[coords_cx[i], coords_cy[i], coords_cz[i]] = -1

        # Make the list of centers from the updated ctr_hmp
        ctr_all = torch.nonzero(ctr_hmp > 0).short()

        if self.top_k is None:
            return ctr_all
        elif ctr_all.size(0) < self.top_k:
            return ctr_all
        else:
            # find top k centers.
            top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), self.top_k)
            return torch.nonzero(ctr_hmp >= top_k_scores[-1]).short()

    def _group_pixels(self, ctr: torch.Tensor, offsets: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Inspired by https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py
        Gives each pixel in the image an instance id.
        Arguments:
            ctr: A Tensor of shape [K, 3] where K is the number of center points. The order of second dim is (z, y, x).
            offsets: A Tensor of shape [N, 3, H, W, D] of raw offset output, where N is the batch size,
                for consistent, we only support N=1. The order of second dim is (offset_z, offset_y, offset_x).
        Returns:
            if self.compute_voting is False:
                A Tensor of shape [1, H, W, D] with instance ids for every voxel in the image.
            if self.compute_voting is True:
                A tuple of:
                    A Tensor of shape [1, H, W, D] with instance ids for every voxel in the image.
                    A Tensor of shape [H, W, D] with the number of votes each voxel got.
        """
        offsets = offsets.squeeze(0)
        depth, height, width = offsets.size()[1:]

        # generates a 3D coordinate map, where each location is the coordinate of that loc
        z_coord, y_coord, x_coord = torch.meshgrid(
            torch.arange(depth),
            torch.arange(height),
            torch.arange(width),
            indexing="ij"
        )
        z_coord = z_coord[None, :].to(offsets.device)
        y_coord = y_coord[None, :].to(offsets.device)
        x_coord = x_coord[None, :].to(offsets.device)

        coord = torch.cat((z_coord, y_coord, x_coord), dim=0)

        ctr_loc = (coord + offsets)
        
        votes = None
        if self.compute_voting:
            votes = torch.round(ctr_loc).long()
            votes[2, :] = torch.clamp(votes[2, :], 0, width - 1)
            votes[1, :] = torch.clamp(votes[1, :], 0, height - 1)
            votes[0, :] = torch.clamp(votes[0, :], 0, depth - 1)

            flat_votes = votes.view(3, -1)
            # Calculate unique coordinate values and their counts
            unique_coords, counts = torch.unique(flat_votes, dim=1, return_counts=True)
            # Create a result tensor with zeros
            votes = torch.zeros(1, votes.shape[1], votes.shape[2], votes.shape[3], dtype=torch.long,
                                device=votes.device)
            # Use advanced indexing to set counts in the result tensor
            votes[0, unique_coords[0], unique_coords[1], unique_coords[2]] = counts

        if ctr.shape[0] == 0:
            if self.compute_voting:
                return torch.zeros(1, depth, height, width), torch.squeeze(votes)
            else:
                return torch.zeros(1, depth, height, width)

        ctr_loc = ctr_loc.view(3, depth * height * width).transpose(1, 0)

        del z_coord, y_coord, x_coord, coord
        torch.cuda.empty_cache()

        # ctr: [K, 3] -> [K, 1, 3]
        # ctr_loc = [D*H*W, 3] -> [1, D*H*W, 3]
        ctr = ctr.unsqueeze(1)
        ctr_loc = ctr_loc.unsqueeze(0)

        # Compute the distances in batches to avoid memory issues
        total_elements = ctr_loc.shape[1]
        batch_size = 1e6
        num_batches = (total_elements + batch_size - 1) // batch_size

        # Initialize a list to store the results for each batch
        instance_id_batches = []

        for batch_idx in range(int(num_batches)):
            start_idx = int(batch_idx * batch_size)
            end_idx = int(min((batch_idx + 1) * batch_size, total_elements))

            # Process a batch of elements
            ctr_loc_batch = ctr_loc[:, start_idx:end_idx]  # Slice along dim=1
            distance_batch = torch.norm(ctr - ctr_loc_batch, dim=-1)  # [K, batch_size]

            # Find the center with the minimum distance at each voxel, offset by 1
            instance_id_batch = torch.argmin(distance_batch, dim=0).short() + 1
            instance_id_batches.append(instance_id_batch)

        # Concatenate the results along the batch dimension
        instance_id = torch.cat(instance_id_batches, dim=0).view(1, depth, height, width)

        if self.compute_voting:
            return instance_id, torch.squeeze(votes)
        return instance_id

    def _calibrate_offsets(self, offsets: torch.Tensor, centers: np.ndarray) -> torch.Tensor:
        """
        Calibrates the offsets by subtracting the mean offset at center locations
        Args:
            offsets: A Tensor of shape [N, 3, W, H, D] of raw offset output, where N is the batch size (N=1 expected)
            centers: Binary np.ndarray of dimension (H, W, D), array of centers
        """
        bias_x, bias_y, bias_z = torch.mean(offsets[:, :, centers == 1], axis=2).squeeze()
        offsets[:, 0, :, :, :] = offsets[:, 0, :, :, :] - bias_x
        offsets[:, 1, :, :, :] = offsets[:, 1, :, :, :] - bias_y
        offsets[:, 2, :, :, :] = offsets[:, 2, :, :, :] - bias_z
        return offsets

    def _postprocess(self, output_dict: Dict[str, NdarrayOrTensor]) -> Dict[str, NdarrayOrTensor]:
        semantic_pred_proba = output_dict['semantic_pred_proba']
        binary_pred = np.squeeze(self._maybe_convert_to_numpy(self.binarize_semantic_probability(semantic_pred_proba)))
        instance_seg_pred = label(binary_pred)[0]
        output_dict['instance_seg_pred'] = self._convert_as(instance_seg_pred, semantic_pred_proba)
        output_dict['semantic_pred_binary'] = self._convert_as(binary_pred, semantic_pred_proba)

        output_dict = self.remove_small_instances(output_dict)
        instance_centers = self._find_instance_center(output_dict['center_pred'])

        centers_mx = np.zeros_like(binary_pred)
        ic = instance_centers.cpu().numpy()
        centers_mx[ic[:, 0], ic[:, 1], ic[:, 2]] = 1

        if self.calibrate_offsets:
            output_dict['offsets_pred'] = self._calibrate_offsets(output_dict['offsets_pred'], centers_mx)
            
        instance_ids = self._group_pixels(instance_centers, output_dict['offsets_pred'])
        if self.compute_voting:
            instance_ids, voting_image = instance_ids
            output_dict['voting_image'] = self._convert_as(voting_image, semantic_pred_proba)

        instance_mask = np.squeeze(instance_ids.cpu().numpy().astype(np.int32)) * binary_pred

        # Converting dtype to float32 since nibabel cannot handle float16
        cp = np.squeeze(self._maybe_convert_to_numpy(output_dict['center_pred']).astype(np.float32))
        output_dict['center_pred'] = self._convert_as(cp, semantic_pred_proba)
        offsets = np.squeeze(self._maybe_convert_to_numpy(output_dict['offsets_pred']).astype(np.float32))
        output_dict['offsets_pred'] = self._convert_as(offsets, semantic_pred_proba)
        output_dict['offsets_pred_x'] = output_dict['offsets_pred'][0]
        output_dict['offsets_pred_y'] = output_dict['offsets_pred'][1]
        output_dict['offsets_pred_z'] = output_dict['offsets_pred'][2]

        output_dict['instance_seg_pred'] = self._convert_as(instance_mask, semantic_pred_proba)
        output_dict['semantic_pred_binary'] = self._convert_as(binary_pred, semantic_pred_proba)
        if type(output_dict['semantic_pred_proba']) == torch.Tensor:
            output_dict['semantic_pred_proba'] = torch.squeeze(output_dict['semantic_pred_proba'])
        else:
            output_dict['semantic_pred_proba'] = np.squeeze(output_dict['semantic_pred_proba'])
        output_dict = self.refine_instance_segmentation(output_dict)

        return output_dict


if __name__ == '__main__':
    from conflunet.utilities.planning_and_configuration import load_dataset_and_configuration
    from conflunet.architecture.conflunet import ConfLUNet, UNet3D
    from conflunet.inference.predictors.instance import ConfLUNetPredictor
    from torch import nn
    import torch
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(321)

    model = ConfLUNet(1,2, scale_offsets=20)
    path_to_model = "/home/mwynen/Downloads/best_DSC_SO_A_L1_e-5_h1200o0.3_S20_seed1.pth"
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    p = ConfLUNetPredictor(
        plans_manager=plans_manager,
        model=model,
        postprocessor=ConfLUNetPostprocessor(
            minimum_instance_size=3,
            semantic_threshold=0.5,
            heatmap_threshold=0.1,
            nms_kernel_size=3,
            top_k=None,
            compute_voting=False,
            calibrate_offsets=False,
            device=torch.device('cuda:0')
        ),
        output_dir='/home/mwynen/data/nnUNet/tmp/output_dir_test',
        preprocessed_files_dir='/home/mwynen/data/nnUNet/tmp/preprocessed_dir',
        num_workers=0,
        save_only_instance_segmentation=False,
    )
    # p.predict_from_preprocessed_dir()

    dataloader = p.get_dataloader()
    predictions_loader = p.get_predictions_loader(dataloader, p.model)

    for data_batch, predicted_batch in zip(dataloader, predictions_loader):
        break
