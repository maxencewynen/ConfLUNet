import numpy as np
from scipy.ndimage import label
from typing import Sequence, Union


def is_too_small(
        instance_segmentation: np.array, 
        lesion_id: int, 
        voxel_size: Tuple[float, float, float] = (1, 1, 1),
        minimum_instance_size: int = 14, 
        minimum_size_along_axis: int = 3
):
    """
    Check if a lesion is too small to be considered a real lesion.
    Args:
        instance_segmentation (np.array): The instance mask.
        lesion_id (int): The id of the lesion to be checked.
        voxel_size (Tuple[float, float, float], optional): The voxel size along each axis.
        minimum_instance_size (int, optional): The minimum size of a lesion in mm^3.
        minimum_size_along_axis (int, optional): The minimum size of a lesion along each axis.
    """
    assert type(voxel_size) == tuple, "Voxel size should be a tuple"
    assert len(voxel_size) == len(instance_segmentation.shape), \
        "Voxel size should be a tuple of same length as the instance segmentation shape tuple"

    this_instance_indices = np.where(instance_segmentation == lesion_id)
    if len(this_instance_indices[0]) == 0:
        return True
    size_along_x = (1 + max(this_instance_indices[0]) - min(this_instance_indices[0])) * voxel_size[0]
    size_along_y = (1 + max(this_instance_indices[1]) - min(this_instance_indices[1])) * voxel_size[1]
    if len(this_instance_indices) == 3:
        size_along_z = (1 + max(this_instance_indices[2]) - min(this_instance_indices[2])) * voxel_size[2]
        # if the connected component is smaller than 3mm in any direction, skip it as it is not
        # clinically considered a lesion
        if size_along_x < minimum_size_along_axis or size_along_y < minimum_size_along_axis or size_along_z < minimum_size_along_axis:
            return True
    elif size_along_x < minimum_size_along_axis or size_along_y < minimum_size_along_axis:
        return True

    return len(this_instance_indices[0]) * np.prod(voxel_size) <= minimum_instance_size


def remove_small_lesions_from_instance_segmentation(
        instance_segmentation: np.ndarray, 
        voxel_size: Tuple[float, float, float],
        minimum_instance_size: int = 14, 
        minimum_size_along_axis: int = 3) -> np.ndarray:
    """
    Remove all lesions with less volume than `minimum_instance_size` from an instance segmentation mask `instance_segmentation`.
    Args:
        instance_segmentation: `numpy.ndarray` of shape (H, W[, D]), with a binary lesions segmentation mask.
        voxel_size: `tuple` with the voxel size in mm.
        minimum_instance_size:  `int`, minimal volume of a lesion.
        minimum_size_along_axis: `int`, minimal size of a lesion along any axis.
    Returns:
        Instance lesion segmentation mask (`numpy.ndarray` of shape (H, W, D]))
    """

    assert type(voxel_size) == tuple, "Voxel size should be a tuple"
    assert len(voxel_size) == len(instance_segmentation.shape), \
        "Voxel size should be a tuple of same length as the instance segmentation shape tuple"

    label_list, label_counts = np.unique(instance_segmentation, return_counts=True)

    instance_seg2 = np.zeros_like(instance_segmentation)

    for lid, lvoxels in zip(label_list, label_counts):
        if lid == 0: continue

        if not is_too_small(instance_segmentation, lid, voxel_size, minimum_instance_size,
                            minimum_size_along_axis=minimum_size_along_axis):
            instance_seg2[instance_segmentation == lid] = lid

    return instance_seg2


def remove_small_lesions_from_binary_segmentation(
        binary_segmentation: np.ndarray, 
        voxel_size: Tuple[int, int, int],
        minimum_instance_size: int = 14, 
        minimum_size_along_axis: int = 3) -> np.ndarray:
    """
    Remove all lesions with less volume than `minimum_instance_size` from a binary segmentation mask `binary_segmentation`.
    Args:
        binary_segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
        voxel_size: `tuple` of length 3, with the voxel size in mm.
        minimum_instance_size:  `int`, minimal volume of a lesion.
        minimum_size_along_axis: `int`, minimal size of a lesion along any axis.
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

    seg2 = np.zeros_like(binary_segmentation)
    for i_el in range(1, num_labels + 1):
        if not is_too_small(labeled_seg, i_el, voxel_size, minimum_instance_size, minimum_size_along_axis=minimum_size_along_axis):
            this_instance_indices = np.where(labeled_seg == i_el)
            this_instance_mask = np.stack(this_instance_indices, axis=1)
            current_voxels = this_instance_mask
            seg2[current_voxels[:, 0],
            current_voxels[:, 1],
            current_voxels[:, 2]] = 1
    return seg2

