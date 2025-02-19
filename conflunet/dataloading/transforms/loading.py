import numpy as np
from pickle import load
from os.path import exists as pexists
from scipy.ndimage import center_of_mass

from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.data.meta_tensor import MetaTensor

from conflunet.postprocessing.small_instances_removal import remove_small_lesions_from_instance_segmentation


class CustomLoadNPZInstanced(MapTransform):
    def __init__(self, keys: KeysCollection, test=False, get_small_instances=False,
                 get_confluent_instances=False, allow_missing_keys=False):
        super().__init__(keys)
        if type(keys) == list and len(keys) > 1:
            raise Exception("This transform should only be used with 1 key.")
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.test = test
        self.get_small_instances = get_small_instances
        self.get_confluent_instances = get_confluent_instances

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            array = np.load(d[key], allow_pickle=False)
            d['img'] = MetaTensor(array['data'].astype(np.float32))
            if not self.test:
                # casting the segmentation in np.float32 otherwise there is a weird collate error with monai
                d['instance_seg'] = MetaTensor(array['instance_seg'].astype(np.float32))
                d['seg'] = MetaTensor((array['seg'] > 0).astype(np.float32))

                # brainmask includes brain and lesions, area around the brain is considered background
                # Not to confuse with other cases (like for RandCropByPosNegLabeld) where the foreground only includes the lesions
                # Notice the >= instead of >, as nnUNet preprocessing makes the actual background to be =-1
                d['brainmask'] = MetaTensor((array['seg'] >= 0).astype(np.float32))

                # if 'center_heatmap' in d.keys():
                #     d['center_heatmap'] = d['center_heatmap'].astype(np.float32)
                if (self.get_small_instances and self.get_confluent_instances and
                        'small_objects_and_confluent_instances_classes' in array.keys()):
                    weights = array['small_objects_and_confluent_instances_classes'].astype(np.float32)
                    weights[weights == 1] = 1  # large, not confluent
                    weights[weights == 2] = 3  # large, confluent
                    weights[weights == 3] = 3  # small, not confluent
                    weights[weights == 4] = 3  # small, confluent
                    d['weights'] = MetaTensor(weights)
                elif self.get_small_instances and 'small_object_classes' in array.keys():
                    weights = array['small_object_classes'].astype(np.float32)
                    weights[weights == 2] = 3  # small
                    d['weights'] = MetaTensor(weights)
                elif self.get_confluent_instances and 'confluent_instances' in array.keys():
                    weights = array['confluent_instances'].astype(np.float32)
                    weights[weights == 2] = 1  # confluent
                    d['weights'] = MetaTensor(weights)
            elif 'seg' in d.keys():
                d['brainmask'] = MetaTensor((d['seg'] >= 0).astype(np.float32))
            elif 'brainmask' in d.keys():
                d['brainmask'] = MetaTensor(d['brainmask'].astype(np.float32))

            properties_file = d['properties_file']
            if not pexists(properties_file):
                raise ValueError(f"Properties file {properties_file} not found")
            with open(properties_file, 'rb') as f:
                properties = load(f)
            d['properties'] = properties
        return d


def make_offset_matrices(data, sigma=2):
    # Define 3D Gaussian function
    def gaussian_3d(x, y, z, cx, cy, cz, sigma):
        return np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * sigma ** 2))

    data = np.squeeze(data)

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

    def __init__(self, keys: KeysCollection, allow_missing_keys=False, sigma=2):
        """
        Args:
            key (str): the key corresponding to the desired data in the dictionary to apply the transformation.
        """
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if type(keys) == list and len(keys) > 1:
            raise Exception("This transform should only be used with 1 key.")
        self.sigma = sigma

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            com_gt, com_reg = self.make_offset_matrices(d[key], sigma=self.sigma)
            d["center_heatmap"] = com_gt
            d["offsets"] = com_reg
            # d["seg"] = (d[key] > 0).astype(np.uint8)
        return d

    def make_offset_matrices(self, data, sigma=2):
        return make_offset_matrices(data, sigma=sigma)


class RemoveSmallInstancesTransform(MapTransform):
    def __init__(self, keys: KeysCollection, minimum_instance_size: int = 14, minimum_size_along_axis: int = 3, voxel_size=(1, 1, 1),
                 instance_seg_key='instance_seg'):
        super().__init__(keys)
        self.keys = keys
        self.minimum_instance_size = minimum_instance_size
        self.minimum_size_along_axis = minimum_size_along_axis
        self.voxel_size = voxel_size
        self.instance_seg_key = instance_seg_key

    def __call__(self, data):
        d = dict(data)
        original_shape = d[self.instance_seg_key].shape
        batch_size = d[self.instance_seg_key].shape[0]
        cleaned_instance_segmentations = []
        for i in range(batch_size):
             cleaned_instance_segmentations.append(
                 remove_small_lesions_from_instance_segmentation(np.squeeze(data[self.instance_seg_key][i]),
                                                                 voxel_size=self.voxel_size,
                                                                 minimum_instance_size=self.minimum_size_along_axis)
             )
        d[self.instance_seg_key] = np.stack(cleaned_instance_segmentations, axis=0)
        assert d[self.instance_seg_key].shape == original_shape, f"Expected shape {original_shape}, got {d[self.instance_seg_key].shape}"

        for key in self.key_iterator(d):
            if key == self.instance_seg_key:
                continue
            if key in d.keys():
                d[key] *= (d[self.instance_seg_key] > 0).astype(np.float32)

        return d
