import nibabel as nib
import numpy as np

from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice
from nnunetv2.preprocessing.cropping.cropping import create_nonzero_mask
from scipy.ndimage import center_of_mass
from conflunet.evaluation.utils import find_confluent_lesions


def load_nii(path, affine=False):
    nib_obj = nib.load(path)
    data = nib_obj.get_fdata()
    if affine:
        return data, nib_obj.affine
    return data


def save_nii(path, data, affine=None):
    if affine is None:
        affine = np.eye(4)
    nib_obj = nib.Nifti1Image(data, affine)
    nib.save(nib_obj, path)


def save_npz_key_as_nii(path, key, affine=None):
    data = np.load(path)
    if key not in data.keys():
        raise ValueError(f'Key {key} not found in {path}')
    data = data[key]
    save_nii(path.replace('.npz', '.nii.gz'), data, affine)


def save_npz_as_nii_files(path, affine=None):
    data = np.load(path)
    for key in data.keys():
        if len(data[key].shape) > 3:
            if data[key].shape[0] == 1:
                save_nii(path.replace('.npz', f'_{key}.nii.gz'), data[key].squeeze(), affine)
            else:
                for i in range(data[key].shape[0]):
                    save_nii(path.replace('.npz', f'_{key}_{i}.nii.gz'), data[key][i], affine)
        else:
            save_nii(path.replace('.npz', f'_{key}.nii.gz'), data[key], affine)


def crop_to_nonzero(data, seg=None, instance_seg=None, nonzero_label=-1):
    """
    Adapted from nnunetv2.preprocessing.cropping.cropping to handle instance segmentation maps
    :param instance_seg:
    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]

    slicer = (slice(None),) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    if instance_seg is not None:
        instance_seg = instance_seg[slicer]
    else:
        # print('Instance segmentation map not provided. Returning None.')
        instance_seg = None
    return data, seg, instance_seg, bbox


def create_center_heatmap_from_instance_seg(instance_seg: np.array, sigma: int = 2):
    """
    Create 3D Gaussian heatmaps centered at the center of mass of each lesion in the instance segmentation map
    :param instance_seg: 3D instance segmentation map
    :param sigma: standard deviation of the Gaussian
    :return:
    """
    # Define 3D Gaussian function
    def gaussian_3d(x, y, z, cx, cy, cz):
        return np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * sigma ** 2))

    instances = np.squeeze(instance_seg)
    heatmap = np.zeros_like(instances, dtype=np.float32)

    # Create coordinate grids
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(instances.shape[0]),
                                         np.arange(instances.shape[1]),
                                         np.arange(instances.shape[2]),
                                         indexing='ij')

    # Get all unique lesion IDs (excluding zero which is typically background)
    lesion_ids = np.unique(instances[instances != 0])

    # For each lesion id
    for lesion_id in lesion_ids:
        # Get binary mask for the current lesion
        mask = (instances == lesion_id)

        # Compute the center of mass of the lesion
        cx, cy, cz = center_of_mass(mask)

        # Compute heatmap values using broadcasting
        current_gaussian = gaussian_3d(x_grid, y_grid, z_grid, cx, cy, cz)

        # Update heatmap with the maximum value encountered so far at each voxel
        heatmap = np.maximum(heatmap, current_gaussian)

    if len(heatmap.shape) < len(instance_seg.shape):
        heatmap = np.expand_dims(heatmap, axis=0)
    return heatmap


def get_small_object_classes(instance_seg: np.array, threshold: int = 100):
    """
    Transforms instance segmentation map into a map of small objects where label=1 if the object is bigger than the
    threshold, label=2 if it is smaller and 0 if background
    :param instance_seg: instance segmentation
    :param threshold: threshold for the size of the object in voxels
    :return: segmentation map with small objects labeled as 2
    """
    instances = np.squeeze(instance_seg)
    small_objects = np.zeros_like(instances, dtype=np.uint8)

    # Get all unique lesion IDs (excluding zero which is typically background)
    lesion_ids, voxel_counts = np.unique(instances[instances != 0], return_counts=True)

    # For each lesion id
    for lesion_id, voxel_count in zip(lesion_ids, voxel_counts):
        if lesion_id == 0: continue # Skip background
        if voxel_count < threshold:
            small_objects[instances == lesion_id] = 2
        else:
            small_objects[instances == lesion_id] = 1

    if len(small_objects.shape) < len(instance_seg.shape):
        small_objects = np.expand_dims(small_objects, axis=0)

    return small_objects.astype(np.uint8)


def get_confluent_instances_classes(instance_seg: np.array, parallel: bool = True):
    """
    Transforms instance segmentation map into a map of confluent instances where label=1 if the instance is not part of
    a larger confluent cluster, 2 if it is part of a larger confluent cluster and 0 if background
    """
    instances = np.squeeze(instance_seg)
    confluent_instances_ids = find_confluent_lesions(instances)

    confluence_map = (instances > 0).astype(np.uint8)

    for lesion_id in confluent_instances_ids:
        confluence_map[instances == lesion_id] = 2

    if len(confluence_map.shape) < len(instance_seg.shape):
        confluence_map = np.expand_dims(confluence_map, axis=0)

    return confluence_map.astype(np.uint8)


def merge_maps(small_objects: np.array, confluent_instances: np.array):
    """
    Merge small objects map and confluent instances map.
    :param small_objects: map of small objects. Label=1 if the object is large, label=2 if it is small
    :param confluent_instances: map of confluent instances. Label=1 if the instance is not confluent, label=2 if it is
    :return: merged map where label=1 if the object is large and not confluent, label=2 if it is large and confluent,
        label=3 if it is small and not confluent, label=4 if it is small and confluent, label=0 if background
    """
    merged_map = np.zeros_like(small_objects, dtype=np.uint8)
    merged_map[(small_objects == 1) & (confluent_instances == 1)] = 1  # large, not confluent
    merged_map[(small_objects == 1) & (confluent_instances == 2)] = 2  # large, confluent
    merged_map[(small_objects == 2) & (confluent_instances == 1)] = 3  # small, not confluent
    merged_map[(small_objects == 2) & (confluent_instances == 2)] = 4  # small, confluent
    return merged_map


if __name__ == '__main__':
    pass
    save_npz_as_nii_files('/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans_3d_fullres/sub-055_ses-01.npz')