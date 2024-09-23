from monai.config import KeysCollection
from monai.transforms import MapTransform
import numpy as np
from os.path import exists as pexists
from pickle import load
from conflunet.postprocess import remove_small_lesions_from_instance_segmentation
from scipy.ndimage import center_of_mass


class CustomLoadNPZInstanced(MapTransform):
    def __init__(self, keys: KeysCollection, test=False, allow_missing_keys=False):
        super().__init__(keys)
        if type(keys) == list and len(keys) > 1:
            raise Exception("This transform should only be used with 1 key.")
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.test = test

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            array = np.load(d[key], allow_pickle=False)
            d['img'] = array['data'].astype(np.float32)
            #d['img'] = (array['seg'] > 0).astype(np.float32)
            if not self.test:
                d['instance_seg'] = array['instance_seg'].astype(
                    np.uint8)  # assuming no patient could have more than 255 lesions
                d['seg'] = (array['seg'] > 0).astype(np.uint8)

                # brainmask includes brain and lesions, area around the brain is considered background
                # Not to confuse with other cases (like for RandCropByPosNegLabeld) where the foreground only includes the lesions
                # Notice the >= instead of >, as nnUNet preprocessing makes the actual background to be =-1
                d['brainmask'] = (array['seg'] >= 0).astype(np.uint8)

                if 'center_heatmap' in d.keys():
                    d['center_heatmap'] = d['center_heatmap'].astype(np.float32)

                if 'small_objects_and_confluent_instances_classes' in d.keys():
                    d['small_objects_and_confluent_instances_classes'] = \
                        d['small_objects_and_confluent_instances_classes'].astype(np.uint8)
                if 'small_object_classes' in d.keys():
                    d['small_object_classes'] = d['small_object_classes'].astype(np.uint8)
                if 'confluent_instances' in d.keys():
                    d['confluent_instances'] = d['confluent_instances'].astype(np.uint8)
            elif 'brainmask' in d.keys():
                d['brainmask'] = d['brainmask'].astype(np.uint8)

            properties_file = d['properties_file']
            if not pexists(properties_file):
                raise ValueError(f"Properties file {properties_file} not found")
            with open(properties_file, 'rb') as f:
                properties = load(f)
            d['properties'] = properties
        return d


def make_offset_matrices(data, sigma=2, voxel_size=(1, 1, 1), remove_small_lesions=False, l_min=14):
    # Define 3D Gaussian function
    def gaussian_3d(x, y, z, cx, cy, cz, sigma):
        return np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * sigma ** 2))

    data = np.squeeze(data)
    if remove_small_lesions:
        data = remove_small_lesions_from_instance_segmentation(data, voxel_size=voxel_size, l_min=l_min)

    heatmap = np.zeros_like(data, dtype=np.float32)
    offset_x = np.zeros_like(data, dtype=np.float32)
    offset_y = np.zeros_like(data, dtype=np.float32)
    offset_z = np.zeros_like(data, dtype=np.float32)

    # Create coordinate grids
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(data.shape[0]),
                                         np.arange(data.shape[1]),
                                         np.arange(data.shape[2]),
                                         indexing='ij')

    # Get all unique lesion IDs (excluding zero which is typically background)
    lesion_ids = np.unique(data[data != 0])

    # For each lesion id
    for lesion_id in lesion_ids:
        # Get binary mask for the current lesion
        mask = (data == lesion_id)

        # Compute the center of mass of the lesion
        cx, cy, cz = center_of_mass(mask)

        # Compute heatmap values using broadcasting
        current_gaussian = gaussian_3d(x_grid, y_grid, z_grid, cx, cy, cz, sigma)
        # Update heatmap with the maximum value encountered so far at each voxel
        heatmap = np.maximum(heatmap, current_gaussian)

        # Update offset matrices
        offset_x[mask] = cx - x_grid[mask]
        offset_y[mask] = cy - y_grid[mask]
        offset_z[mask] = cz - z_grid[mask]

    return np.expand_dims(heatmap, axis=0).astype(np.float32), \
        np.stack([offset_x, offset_y, offset_z], axis=0).astype(np.float32)


class LesionOffsetTransformd(MapTransform):
    """
    A MONAI transform to compute the offsets for each voxel from the center of mass of its lesion.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys=False, remove_small_lesions=False, l_min=14, sigma=2):
        """
        Args:
            key (str): the key corresponding to the desired data in the dictionary to apply the transformation.
        """
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if type(keys) == list and len(keys) > 1:
            raise Exception("This transform should only be used with 1 key.")
        self.remove_small_lesions = remove_small_lesions
        self.l_min = l_min
        self.sigma = sigma

    def __call__(self, data):
        d = dict(data)
        voxel_size = tuple(data['properties']['sitk_stuff']['spacing'])
        for key in self.key_iterator(d):
            com_gt, com_reg = self.make_offset_matrices(d[key], voxel_size=voxel_size, sigma=self.sigma)
            d["center_heatmap"] = com_gt
            d["offsets"] = com_reg
            # d["seg"] = (d[key] > 0).astype(np.uint8)
        return d

    def make_offset_matrices(self, data, sigma=2, voxel_size=(1, 1, 1)):
        return make_offset_matrices(data,
                                    sigma=sigma,
                                    voxel_size=voxel_size,
                                    remove_small_lesions=self.remove_small_lesions,
                                    l_min=self.l_min)
