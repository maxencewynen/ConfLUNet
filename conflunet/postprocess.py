import numpy as np
from scipy.ndimage import maximum_filter, generate_binary_structure, label, labeled_comprehension
import argparse
import torch
import torch.nn.functional as F
from typing import Tuple, Union
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


def compute_hessian_eigenvalues(image, sigma=1):
    """
    Compute the eigenvalues of the Hessian matrix of an image.
    Args:
        image (np.ndarray): Input image.
        sigma (float): Standard deviation of the Gaussian filter used for the Hessian matrix.
    Returns:
        np.ndarray: Eigenvalues of the Hessian matrix.
    """
    hessian_matrices = hessian_matrix(image, sigma=1, use_gaussian_derivatives=False)
    eigs = hessian_matrix_eigvals(hessian_matrices)
    return eigs


def postprocess_probability_segmentation(probability_segmentation: np.ndarray, threshold: float = 0.5,
                                         voxel_size: Tuple[float, float, float] = (1., 1., 1.),
                                         l_min: int = 14) -> np.ndarray:
    """
    Constructs an instance segmentation mask from a lesion probability matrix, by applying a threshold
     and removing all lesions with less volume than `l_min`.
    Args:
        probability_segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
        threshold: `float`, threshold to apply to the binary segmentation mask.
        voxel_size: `tuple` of length 3, with the voxel size in mm.
        l_min:  `int`, minimal volume of a lesion.
    Returns:
        Instance lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
    """
    assert type(probability_segmentation) == np.ndarray, "Probability segmentation should be a numpy array"
    assert 0 <= threshold <= 1, "Threshold should be between 0 and 1"
    assert type(voxel_size) == tuple, "Voxel size should be a tuple"
    assert len(voxel_size) == 3, "Voxel size should be a tuple of length 3"
    assert type(l_min) == int, "l_min should be an integer"

    # Threshold the image
    binary_data = np.where(probability_segmentation >= threshold, 1, 0).astype(np.uint8)

    # Remove objects smaller than l_min voxels
    binary_data = remove_small_lesions_from_binary_segmentation(binary_data, voxel_size=voxel_size, l_min=l_min)

    # Find connected components larger than l_min voxels
    labeled_array, num_features = label(binary_data)

    return labeled_array.astype(np.uint16)


def remove_small_lesions_from_instance_segmentation(instance_segmentation: np.ndarray, voxel_size: Tuple[float, float, float],
                                                    l_min: int = 14) -> np.ndarray:
    """
    Remove all lesions with less volume than `l_min` from an instance segmentation mask `instance_segmentation`.
    Args:
        instance_segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
        voxel_size: `tuple` of length 3, with the voxel size in mm.
        l_min:  `int`, minimal volume of a lesion.
    Returns:
        Instance lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
    """

    assert type(voxel_size) == tuple, "Voxel size should be a tuple"
    assert len(voxel_size) == 3, "Voxel size should be a tuple of length 3"

    label_list, label_counts = np.unique(instance_segmentation, return_counts=True)

    instance_seg2 = np.zeros_like(instance_segmentation)

    for lid, lvoxels in zip(label_list, label_counts):
        if lid == 0: continue

        this_instance_indices = np.where(instance_segmentation == lid)
        size_along_x = (1 + max(this_instance_indices[0]) - min(this_instance_indices[0])) * voxel_size[0]
        size_along_y = (1 + max(this_instance_indices[1]) - min(this_instance_indices[1])) * voxel_size[1]
        size_along_z = (1 + max(this_instance_indices[2]) - min(this_instance_indices[2])) * voxel_size[2]

        # if the connected component is smaller than 3mm in any direction, skip it as it is not
        # clinically considered a lesion
        if size_along_x < 3 or size_along_y < 3 or size_along_z < 3:
            continue

        if lvoxels * np.prod(voxel_size) > l_min:
            instance_seg2[instance_segmentation == lid] = lid

    return instance_seg2


def remove_small_lesions_from_binary_segmentation(binary_segmentation: np.ndarray, voxel_size: Tuple[int, int, int],
                                                  l_min: int = 14) -> np.ndarray:
    """
    Remove all lesions with less volume than `l_min` from a binary segmentation mask `binary_segmentation`.
    Args:
        binary_segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
        voxel_size: `tuple` of length 3, with the voxel size in mm.
        l_min:  `int`, minimal volume of a lesion.
    Returns:
        Binary lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
    """

    assert type(voxel_size) == tuple, "Voxel size should be a tuple"
    assert len(voxel_size) == 3, "Voxel size should be a tuple of length 3"
    unique_values = np.unique(binary_segmentation)
    assert (len(unique_values) == 1 and unique_values[0] == 0) or (
                len(unique_values) == 2 and set(unique_values) == {0, 1}), \
        f"Segmentation should be {0, 1} but got {unique_values}"

    labeled_seg, num_labels = label(binary_segmentation)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = labeled_comprehension(binary_segmentation, labeled_seg, label_list, np.sum, float, 0)

    seg2 = np.zeros_like(binary_segmentation)
    for i_el, n_el in enumerate(num_elements_by_lesion):
        this_instance_indices = np.where(labeled_seg == i_el)
        this_instance_mask = np.stack(this_instance_indices, axis=1)

        size_along_x = (1 + max(this_instance_indices[0]) - min(this_instance_indices[0])) * voxel_size[0]
        size_along_y = (1 + max(this_instance_indices[1]) - min(this_instance_indices[1])) * voxel_size[1]
        size_along_z = (1 + max(this_instance_indices[2]) - min(this_instance_indices[2])) * voxel_size[2]

        # if the connected component is smaller than 3 voxels in any direction, skip it as it is not
        # clinically considered a lesion
        if size_along_x < 3 or size_along_y < 3 or size_along_z < 3:
            continue

        lesion_size = n_el * np.prod(voxel_size)
        if lesion_size > l_min:
            current_voxels = this_instance_mask
            seg2[current_voxels[:, 0],
            current_voxels[:, 1],
            current_voxels[:, 2]] = 1
    return seg2


def find_instance_center(ctr_hmp: torch.Tensor, threshold: float = 0.1, nms_kernel: int = 3, top_k: int = 100) -> torch.Tensor:
    """
    Inspired from https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py
    Find the center points from the center heatmap.
    Arguments:
        ctr_hmp: A Tensor of shape [N, 1, H, W, D] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep. If None, all centers > threshold are kept
    Returns:
        A Tensor of shape [K, 3] where K is the number of center points. The order of second dim is (x, y, z).
    """
    assert 0 <= threshold <= 1, "Threshold should be between 0 and 1"
    assert ctr_hmp.size(0) == 1, "Only supports inference for batch size = 1"
    assert ctr_hmp.size(1) == 1, "Center heatmap should have only one channel"
    assert top_k is None or top_k > 0, "top_k should be None or a positive integer"
    assert len(ctr_hmp.size()) == 5, "Center heatmap should have 5 dimensions"
    assert (nms_kernel % 2) == 1, "NMS kernel must be odd"

    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    # thresholding, setting values below threshold to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1)

    # NMS
    nms_padding = (nms_kernel - 1) // 2
    ctr_hmp_max_pooled = F.max_pool3d(ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding)
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

    if top_k is None:
        return ctr_all
    elif ctr_all.size(0) < top_k:
        return ctr_all
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), top_k)
        return torch.nonzero(ctr_hmp >= top_k_scores[-1]).short()


def find_instance_centers_acls(probability_map: np.ndarray, semantic_mask: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Computes the lesion centers using acls's method
    Arguments:
        probability_map: A numpy.ndarray or a torch.Tensor of shape [W, H, D] of raw probability map output
        semantic_mask: A numpy.ndarray or a torch.Tensor of shape [W, H, D] of raw semantic mask output
        device: A string, the device to use
    Returns:
        A Tensor of shape [K, 3] where K is the number of center points. The order of second dim is (x, y, z).
    """
    if type(probability_map) == torch.Tensor:
        probability_map = probability_map.cpu().numpy()

    assert type(probability_map) == np.ndarray or type(
        probability_map) == torch.Tensor, "Probability map should be a numpy array or a torch tensor"
    assert type(semantic_mask) == np.ndarray or type(
        semantic_mask) == torch.Tensor, "Semantic mask should be a numpy array or a torch tensor"
    assert len(probability_map.shape) == 3, "Probability map and semantic mask should have 3 dimensions"
    assert len(semantic_mask.shape) == 3, "Semantic map and semantic mask should have 3 dimensions"
    assert probability_map.shape == semantic_mask.shape, "Probability map and semantic mask should have the same shape"

    mask = semantic_mask == 1
    masked_image_data = np.where(mask, probability_map, 0)

    eigenvalues = compute_hessian_eigenvalues(masked_image_data)
    lesion_centers_mask = np.all(eigenvalues < 0, axis=0)

    lesion_clusters, n_clusters = label(lesion_centers_mask)

    centers = []
    for c in range(1, n_clusters + 1):
        coords = np.where(lesion_clusters == c)
        coords = np.stack(coords, axis=1)
        coords = np.round(np.mean(coords, axis=0)).astype(np.int16)
        centers.append(coords)

    centers = torch.from_numpy(np.stack(centers, axis=0)).short().to(device)

    return centers


def make_votes_readable(votes):
    votes = torch.log(votes + 1, out=torch.zeros_like(votes, dtype=torch.float32))
    votes = F.avg_pool3d(votes, kernel_size=3, stride=1, padding=1)
    return votes * 100


def group_pixels(ctr: torch.Tensor, offsets: torch.Tensor, compute_voting: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Inspired by https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py
    Gives each pixel in the image an instance id.
    Arguments:
        ctr: A Tensor of shape [K, 3] where K is the number of center points. The order of second dim is (z, y, x).
        offsets: A Tensor of shape [N, 3, H, W, D] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_z, offset_y, offset_x).
        compute_voting: A Boolean, whether to compute the votes image.
    Returns:
        if compute_voting is False:
            A Tensor of shape [1, H, W, D] with instance ids for every voxel in the image.
        if compute_voting is True:
            A tuple of:
                A Tensor of shape [1, H, W, D] with instance ids for every voxel in the image.
                A Tensor of shape [H, W, D] with the number of votes each voxel got.
    """
    assert type(ctr) == torch.Tensor, 'Center points should be a tensor'
    assert type(offsets) == torch.Tensor, 'Offsets should be a tensor'
    assert offsets.size(0) == 1, 'Only supports inference for batch size = 1'
    assert len(ctr.size()) == 2, 'Center points should have 2 dimensions'
    assert len(offsets.size()) == 5, 'Offsets should have 5 dimensions'
    assert type(compute_voting) == bool, 'Compute voting should be a boolean'
    assert ctr.size(0) == 0 or ctr.size(1) == 3, 'Center points should have 3 coordinates'

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

    ctr_loc = (coord + offsets).half()

    if compute_voting:
        votes = torch.round(ctr_loc).long()
        votes[2, :] = torch.clamp(votes[2, :], 0, width - 1)
        votes[1, :] = torch.clamp(votes[1, :], 0, height - 1)
        votes[0, :] = torch.clamp(votes[0, :], 0, depth - 1)

        flat_votes = votes.view(3, -1)
        # Calculate unique coordinate values and their counts
        unique_coords, counts = torch.unique(flat_votes, dim=1, return_counts=True)
        # Create a result tensor with zeros
        votes = torch.zeros(1, votes.shape[1], votes.shape[2], votes.shape[3], dtype=torch.long, device=votes.device)
        # Use advanced indexing to set counts in the result tensor
        votes[0, unique_coords[0], unique_coords[1], unique_coords[2]] = counts

    if ctr.shape[0] == 0:
        if compute_voting:
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

    if compute_voting:
        return instance_id, torch.squeeze(votes)
    return instance_id


def refine_instance_segmentation(instance_mask: np.ndarray, l_min: int = 14) -> np.ndarray:
    """
    Refines the instance segmentation by relabeling disconnected components in instances
    and removing instances smaller than l_min
    Args:
        instance_mask: np.ndarray of dimension (H,W,D), array of instance ids
        l_min: minimum lesion size
    """
    iids = np.unique(instance_mask[instance_mask != 0])
    max_instance_id = np.max(instance_mask)
    # for every instance id
    for iid in iids:
        # get the mask
        mask = (instance_mask == iid)
        components, n_components = label(mask)
        if n_components > 1:  # if the lesion is split in n components
            biggest_lesion_size = 0
            biggest_lesion_id = -1
            for cid in range(1, n_components + 1):  # go through each component
                component_mask = (components == cid)
                this_size = np.sum(component_mask)
                if this_size > biggest_lesion_size:
                    biggest_lesion_size = this_size
                    biggest_lesion_id = cid
            for cid in range(1, n_components + 1):
                if cid == biggest_lesion_id: continue
                instance_mask[components == cid] = 0

        elif np.sum(mask) < l_min:  # check if lesion size is too small or not
            instance_mask[mask] = 0
            instance_mask[instance_mask == max_instance_id] = iid
            max_instance_id -= 1
    return instance_mask


def calibrate_offsets(offsets: torch.Tensor, centers: np.ndarray) -> torch.Tensor:
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


def postprocess(semantic_mask: np.ndarray,
                heatmap: torch.Tensor,
                offsets: torch.Tensor,
                compute_voting: bool = False,
                heatmap_threshold: float = 0.1,
                voxel_size: tuple = (1, 1, 1),
                l_min: int = 14,
                probability_map: np.ndarray = None) -> tuple:
    """
    Postprocesses the semantic mask, center heatmap, and the offsets.

    Args:
        semantic_mask: A binary numpy.ndarray of shape [W, H, D].
        heatmap: A Tensor of shape [N, 1, W, H, D] of raw center heatmap output.
        offsets: A Tensor of shape [N, 3, W, H, D] of raw offset output.
        compute_voting: A Boolean, whether to compute the votes image.
        heatmap_threshold: A Float, threshold applied to the center heatmap score.
        voxel_size: A tuple of length 3, with the voxel size in mm.
        l_min: An Integer, minimal volume of a lesion.
        probability_map: A numpy.ndarray of shape [W, H, D] of raw probability map output.

    Returns:
        tuple: A tuple containing:
            - semantic_mask: A binary numpy.ndarray of shape [W, H, D].
            - instance_mask: A numpy.ndarray of shape [W, H, D].
            - instance_centers: A numpy.ndarray of shape [W, H, D].
            - (Optional) voting_image: A numpy.ndarray of shape [W, H, D].
    """
    assert 0 <= heatmap_threshold <= 1, "Threshold should be between 0 and 1"
    unique_values = np.unique(semantic_mask)
    assert (len(unique_values) == 1 and unique_values[0] == 0) or (
                len(unique_values) == 2 and set(unique_values) == {0, 1}), \
        f"Segmentation should be {0, 1} but got {unique_values}"
    assert type(voxel_size) == tuple, "Voxel size should be a tuple"
    assert len(voxel_size) == 3, "Voxel size should be a tuple of length 3"
    assert type(l_min) == int, "l_min should be an integer"
    assert type(compute_voting) == bool, "Compute voting should be a boolean"
    assert type(
        probability_map) == np.ndarray or probability_map is None, "Probability map should be a numpy array or None"

    semantic_mask = remove_small_lesions_from_binary_segmentation(semantic_mask, voxel_size=voxel_size, l_min=l_min)

    if probability_map is not None:
        instance_centers = find_instance_centers_acls(probability_map, semantic_mask, device=offsets.device)
    else:
        instance_centers = find_instance_center(heatmap, threshold=heatmap_threshold)

    centers_mx = np.zeros_like(semantic_mask)
    ic = instance_centers.cpu().numpy()
    centers_mx[ic[:, 0], ic[:, 1], ic[:, 2]] = 1

    offsets = calibrate_offsets(offsets, centers_mx)

    instance_ids = group_pixels(instance_centers, offsets, compute_voting=compute_voting)
    if compute_voting:
        instance_ids, voting_image = instance_ids
    else:
        voting_image = None

    instance_mask = np.squeeze(instance_ids.cpu().numpy().astype(np.int32)) * semantic_mask
    instance_mask = remove_small_lesions_from_instance_segmentation(instance_mask, voxel_size=voxel_size, l_min=l_min)
    instance_mask = refine_instance_segmentation(instance_mask, l_min=l_min)

    ret = (instance_mask, centers_mx.astype(np.uint8))
    ret += (voting_image.cpu().numpy().astype(np.int16),) if compute_voting else ()
    return (semantic_mask,) + ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get all command line arguments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pred_path', help="Path to the predictions")
    parser.add_argument('--acls', help="Use acls to find instance centers", action="store_true", default=False)
    parser.add_argument('--minimum_lesion_size', help="Minimum lesion size", type=int, default=14)
    parser.add_argument('--heatmap_threshold',
                        help="Probability threshold for a voxel to be considered as a center in the heatmap",
                        type=float, default=0.1)
    parser.add_argument('--compute_voting', help="Compute the voting image", action="store_true", default=False)
    parser.add_argument('--save', help="Whether to save the images or not", action="store_true", default=False)
    parser.add_argument('--save_path', help="File where to save the metrics", default="")
    parser.add_argument('--compute_metrics', help="Compute the metrics", action="store_true", default=False)
    parser.add_argument('--test', help="If the predictions are on the test set", action="store_true", default=False)
    parser.add_argument('--ref_path', help="Path to the predictions", default="/dir/scratchL/mwynen/data/cusl_wml")

    args = parser.parse_args()

    import nibabel as nib
    import os
    from monai.data import write_nifti

    if args.compute_metrics:
        from metrics import *

        metrics_dict = {"Subject_ID": [], "File": []}
        metrics_dict["DSC"] = []
        metrics_dict["PQ"] = []
        metrics_dict["Fbeta"] = []
        metrics_dict["recall"] = []
        metrics_dict["precision"] = []
        metrics_dict["Dice_Per_TP"] = []
        metrics_dict["Pred_Lesion_Count"] = []
        metrics_dict["Ref_Lesion_Count"] = []
        metrics_dict["DiC"] = []
        metrics_dict["CLR"] = []
        metrics_dict["Dice_Per_TP_CL"] = []
        metrics_dict["CL_Count"] = []

        dd = "test" if args.test else "val"
        ref_dir = os.path.join(args.ref_path, dd, "labels")

    # Load the predictions
    for filename in sorted(os.listdir(args.pred_path)):
        if not filename.endswith('seg-binary.nii.gz'):
            continue
        print(f"Processing {filename[:7]}")
        file = os.path.join(args.pred_path, filename)
        binary_seg = nib.load(file)
        voxel_size = binary_seg.header.get_zooms()
        affine = binary_seg.affine
        binary_seg = binary_seg.get_fdata()

        offsets = nib.load(file.replace('seg-binary', 'pred-offsets')).get_fdata()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if offsets.shape[-1] == 3:
            offsets = np.transpose(offsets, (3, 0, 1, 2))
        offsets = torch.from_numpy(offsets).unsqueeze(0).to(device)

        probability_map = nib.load(file.replace('seg-binary', 'pred_prob')).get_fdata() if args.acls else None
        heatmap = torch.from_numpy(nib.load(file.replace('seg-binary', 'pred-heatmap')).get_fdata()).unsqueeze(
            0).unsqueeze(0).to(device) if not args.acls else None

        ret = postprocess(binary_seg, heatmap, offsets, compute_voting=args.compute_voting,
                          heatmap_threshold=args.heatmap_threshold,
                          voxel_size=voxel_size, l_min=args.minimum_lesion_size, probability_map=probability_map)

        spatial_shape = binary_seg.shape

        if args.compute_voting and args.save:
            seg, instances_pred, instance_centers, voting_image = ret
            filename = filename[:14] + "_voting-image.nii.gz"
            filepath = os.path.join(args.pred_path, filename)
            write_nifti(voting_image * seg, filepath,
                        affine=affine,
                        target_affine=affine,
                        output_spatial_shape=spatial_shape)
        else:
            seg, instances_pred, instance_centers = ret

        if args.save:
            # obtain and save predicted offsets
            filename = filename[:14] + "_pred-centers.nii.gz"
            filepath = os.path.join(args.pred_path, filename)
            write_nifti(instance_centers, filepath,
                        affine=affine,
                        target_affine=affine,
                        output_spatial_shape=spatial_shape)

            # obtain and save predicted offsets
        filename = filename[:14] + "_pred-instances.nii.gz"
        filepath = os.path.join(args.pred_path, filename)
        if args.save:
            write_nifti(instances_pred, filepath,
                        affine=affine,
                        target_affine=affine,
                        output_spatial_shape=spatial_shape)

        if not args.compute_metrics:
            continue

        subj_id = filename.split("_ses")[0].split("sub-")[-1]  # Extracting subject ID
        ref_img = nib.load(os.path.join(ref_dir, filename.replace("pred", "mask")))

        voxel_size = ref_img.header.get_zooms()
        ref_img = remove_small_lesions_from_instance_segmentation(ref_img.get_fdata(), voxel_size,
                                                                  l_min=args.minimum_lesion_size)
        pred_img = instances_pred

        matched_pairs, unmatched_pred, unmatched_ref = match_instances(pred_img, ref_img)

        metrics_dict["Subject_ID"].append(subj_id)
        metrics_dict["File"].append(filename)

        dsc = dice_metric((ref_img > 0).astype(np.uint8), (pred_img > 0).astype(np.uint8))
        metrics_dict["DSC"].append(dsc)

        pq_val = panoptic_quality(pred=pred_img, ref=ref_img,
                                  matched_pairs=matched_pairs, unmatched_pred=unmatched_pred,
                                  unmatched_ref=unmatched_ref)
        metrics_dict["PQ"].append(pq_val)
        fbeta_val = f_beta_score(matched_pairs=matched_pairs, unmatched_pred=unmatched_pred,
                                 unmatched_ref=unmatched_ref)
        metrics_dict["Fbeta"].append(fbeta_val)

        recall_val = recall(matched_pairs=matched_pairs, unmatched_ref=unmatched_ref)
        metrics_dict["recall"].append(recall_val)

        precision_val = precision(matched_pairs=matched_pairs, unmatched_pred=unmatched_pred)
        metrics_dict["precision"].append(precision_val)
        dice_scores = dice_per_tp(pred_img, ref_img, matched_pairs)
        # Assuming you want the average Dice score per subject
        avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
        metrics_dict["Dice_Per_TP"].append(avg_dice)

        metrics_dict["Pred_Lesion_Count"].append(pred_lesion_count(pred_img))
        metrics_dict["Ref_Lesion_Count"].append(ref_lesion_count(ref_img))
        metrics_dict["DiC"].append(DiC(pred_img, ref_img))

        confluents_ref_img = np.copy(ref_img)
        cl_ids = find_confluent_lesions(confluents_ref_img)

        # set all other ids to 0 in ref_img
        for id in np.unique(confluents_ref_img):
            if id not in cl_ids:
                confluents_ref_img[confluents_ref_img == id] = 0

        matched_pairs_cl, unmatched_pred_cl, unmatched_ref_cl = match_instances(pred_img, confluents_ref_img)

        clm = len(cl_ids)
        metrics_dict["CL_Count"].append(clm)

        if clm == 0:
            metrics_dict["CLR"].append(np.nan)
        else:
            clr = recall(matched_pairs=matched_pairs_cl, unmatched_ref=unmatched_ref_cl)
            metrics_dict["CLR"].append(clr)

        if clm == 0:
            metrics_dict["Dice_Per_TP_CL"].append(np.nan)
        else:
            dice_scores = dice_per_tp(pred_img, confluents_ref_img, matched_pairs_cl)
            # Assuming you want the average Dice score per subject
            avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
            metrics_dict["Dice_Per_TP_CL"].append(avg_dice)

    if args.compute_metrics:
        df = pd.DataFrame(metrics_dict)
        if args.save_path == "":
            model_name = os.path.basename(os.path.dirname(args.pred_path))
            # Convert dictionary to dataframe and save as CSV
            save_path = os.path.join(args.pred_path, f"metrics_{model_name}_{dd}.csv")
        else:
            save_path = args.save_path
        df.to_csv(save_path, index=False)
