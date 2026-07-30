import warnings

import numpy as np

from monai.transforms import (
    ToTensord, RandFlipd, RandScaleIntensityd, FgBgToIndicesd, RandGaussianNoised, RandGaussianSmoothd)
import numpy as np
import torch
from typing import Callable, Tuple, Sequence, Mapping, Hashable, Any, Union
from copy import deepcopy
from monai.data.meta_tensor import MetaTensor
import torch.nn.functional as F
import os
import nibabel as nib
from monai.transforms import MapTransform
from monai.config import KeysCollection
from copy import copy

from conflunet.postprocess import remove_small_lesions_from_instance_segmentation
from scipy.ndimage import center_of_mass
from os.path import exists as pexists
from pickle import load
from monai.transforms import (
    Compose,
    RandAffined,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
)
from conflunet.dataloading.scaleintensityfixedmean import RandScaleIntensityFixedMeand
from conflunet.dataloading.adjustcontrast import RandAdjustContrastd
from conflunet.dataloading.simulatelowresolution import RandSimulateLowResolutiond



def get_nnunet_spatial_transforms(image_key: str = "img",
                                  seg_keys: list = ["seg", "instance_seg"]):
    keys = [image_key] + seg_keys
    interpolation = ("bilinear",) + ("nearest",) * len(seg_keys)
    transforms = []
    transforms.append(
        RandAffined(
            keys=keys, mode=interpolation, prob=0.2,
            rotate_range=(2 * np.pi, ) * 3,
            scale_range=((-0.3, 0.4),) * 3,
            padding_mode="zeros",
        )
    )
    return transforms


def get_nnunet_augmentations(image_key: str = "img", seg_keys: list = ["seg", "instance_seg"]):
    keys = [image_key] + seg_keys

    transforms = []
    transforms.append(
        RandGaussianNoised(
            keys=image_key, mean=0, std=0.1, prob=0.15
        )
    )

    transforms.append(
        RandGaussianSmoothd(
            keys=image_key, sigma_x=(0.5, 1.), sigma_y=(0.5, 1.), sigma_z=(0.5, 1.), prob=0.2
        )
    )

    transforms.append(
        RandScaleIntensityd(
            keys=image_key, factors=(-0.3, 0.3), prob=0.15
        )
    )

    # This should be equivalent to batchgeneratorsv2.transforms.intensity.contrast.ContrastTransform
    # Update to later version of monai for this to work!
    transforms.append(
        RandScaleIntensityFixedMeand(
            keys=image_key, factors=(-0.25, 0.25), prob=0.15
        )
    )

    # This should be equivalent to batchgeneratorsv2.transforms.spatial.SimulateLowResolutionTransform
    transforms.append(
        RandSimulateLowResolutiond(
             # downsample by NN interpolation and upsample by cubic interpolation
            keys=image_key, downsample_mode="nearest", upsample_mode="trilinear", prob=0.25,
            zoom_range=(0.5, 1.0), align_corners=True
        )
    )

    # This should be equivalent to batchgeneratorsv2.transforms.intensity.gamma.GammaTransform
    # In nnUNet, GammaTransform is called twice with p_invert_image=0 and 1
    # https://github.com/MIC-DKFZ/nnUNet/blob/fee8c2db4a52405389eb5d3c4512bd2f654ab999/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L29
    transforms.append(
        RandAdjustContrastd(
            keys=image_key, gamma=(-0.3, 0.5), prob=0.15,
            invert_image=True, retain_stats=True
        )
    )
    transforms.append(
        RandAdjustContrastd(
            keys=image_key, gamma=(-0.3, 0.5), prob=0.3,
            invert_image=False, retain_stats=True
        )
    )

    # Random mirroring
    transforms.append(
        RandFlipd(keys=keys, spatial_axis=(0, 1, 2), prob=0.5,),
    )

    return transforms


def get_train_transforms(seed: Union[int, None] = None,
                         initial_patch_size: tuple = (128, 128, 128),
                         patch_size: tuple = (96, 96, 96)):
    # if any([x > 128 for x in patch_size]):
    #     raise ValueError(f"patch_size too large. expected a patch size smaller than (128, 128, 128)"
    #                      f"but got {patch_size}")
    transform_list = [
        CustomLoadNPZInstanced(keys=['data']),
        FgBgToIndicesd(keys=['seg']),
        # Crop random fixed sized regions with the center being a foreground or background voxel
        # based on the Pos Neg Ratio.
        RandCropByPosNegLabeld(keys=['img', 'seg', 'instance_seg', 'brainmask'],
                               label_key="seg",
                               fg_indices_key="seg_fg_indices",
                               bg_indices_key="seg_bg_indices",
                               spatial_size=initial_patch_size, num_samples=1,
                               pos=2, neg=1),
        *get_nnunet_spatial_transforms(image_key="img", seg_keys=["seg", "instance_seg", 'brainmask']),
        RandSpatialCropd(keys=['img', 'seg', 'instance_seg', 'brainmask'],
                         roi_size=patch_size,
                         random_center=False, random_size=False),
        *get_nnunet_augmentations(image_key="img", seg_keys=["seg", "instance_seg", 'brainmask']),
        LesionOffsetTransformd(keys="instance_seg"),
        ToTensord(keys=['img', 'seg', 'offsets', 'center_heatmap', 'brainmask']),
        DeleteKeysd(keys=['properties']),
    ]
    transform = Compose(transform_list)

    if seed is not None:
        transform.set_random_state(seed=seed)
    return transform


def get_OLD_train_transforms(seed: Union[int, None] = None,
                         initial_patch_size: tuple = (128, 128, 128),
                         patch_size: tuple = (96, 96, 96)):
    # if any([x > 128 for x in patch_size]):
    #     raise ValueError(f"patch_size too large. expected a patch size smaller than (128, 128, 128)"
    #                      f"but got {patch_size}")
    transform_list = [
        CustomLoadNPZInstanced(keys=['data']),
        FgBgToIndicesd(keys=['brainmask']),
        # Crop random fixed sized regions with the center being a foreground or background voxel
        # based on the Pos Neg Ratio.
        RandCropByPosNegLabeld(keys=['img', 'seg', 'instance_seg', 'brainmask'],
                               label_key="seg",
                               fg_indices_key="brainmask_fg_indices",
                               bg_indices_key="brainmask_bg_indices",
                               spatial_size=initial_patch_size, num_samples=32,
                               # spatial_size=(128, 128, 128), num_samples=32,
                               pos=4, neg=1),
        *get_nnunet_spatial_transforms(image_key="img", seg_keys=["seg", "instance_seg", 'brainmask']),
        RandSpatialCropd(keys=['img', 'seg', 'instance_seg', 'brainmask'],
                         roi_size=patch_size,
                         random_center=True, random_size=False),
        *get_nnunet_augmentations(image_key="img", seg_keys=["seg", "instance_seg", 'brainmask']),
        LesionOffsetTransformd(keys="instance_seg"),
        ToTensord(keys=['img', 'seg', 'offsets', 'center_heatmap', 'brainmask']),
        DeleteKeysd(keys=['properties']),
    ]
    transform = Compose(transform_list)

    if seed is not None:
        transform.set_random_state(seed=seed)
    return transform


def get_val_transforms(seed: Union[int, None] = None,
                       initial_patch_size: tuple = (128, 128, 128),
                       patch_size: tuple = (96, 96, 96)):
    # if any([x > 128 for x in patch_size]):
    #     raise ValueError(f"patch_size too large. expected a patch size smaller than (128, 128, 128)"
    #                      f"but got {patch_size}")
    transform_list = [
        CustomLoadNPZInstanced(keys=['data']),
        FgBgToIndicesd(keys=['brainmask']),
        # Crop random fixed sized regions with the center being a foreground or background voxel
        # based on the Pos Neg Ratio.
        RandCropByPosNegLabeld(keys=['img', 'seg', 'instance_seg', 'brainmask'],
                               label_key="seg",
                               fg_indices_key="brainmask_fg_indices",
                               bg_indices_key="brainmask_bg_indices",
                               spatial_size=initial_patch_size, num_samples=32,
                               # spatial_size=(128, 128, 128), num_samples=32,
                               pos=4, neg=1),
        RandSpatialCropd(keys=['img', 'seg', 'instance_seg', 'brainmask'],
                         roi_size=patch_size,
                         random_center=True, random_size=False),
        LesionOffsetTransformd(keys="instance_seg"),
        ToTensord(keys=['img', 'seg', 'offsets', 'center_heatmap', 'brainmask']),
        DeleteKeysd(keys=['properties']),
    ]
    transform = Compose(transform_list)

    if seed is not None:
        transform.set_random_state(seed=seed)
    return transform


def get_test_transforms():
    transform_list = [
        CustomLoadNPZInstanced(keys=['data'], test=True),
        ToTensord(keys=['img', 'brainmask'], allow_missing_keys=True),
    ]
    transform = Compose(transform_list)
    return transform


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


class DeleteKeysd(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys=True):
        super().__init__(keys)
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data):
        d = dict(data)
        d1 = copy(d)
        for key in self.key_iterator(d):
            if key in self.keys:
                del d1[key]
        return d1


class Printer(Callable):
    """
    A callable class that allows to print a message. Mostly used for debugging
    This class can be used as a callback function in various contexts to print information about the input data.
    Args:
        string (str): A string to be prepended to the printed output.
    Attributes:
        string (str): A string to be prepended to the printed output.
    Returns:
        The input argument, unchanged.

    Example:
        >>> printer = Printer('Received input:')
        >>> _ = printer('Hello World!')
        Received input: Hello World!
        >>> import torch
        >>> _ = printer(torch.tensor([1, 2, 3]))
        Received input: (Shape = torch.Size([3]) )
    """

    def __init__(self, string: str):
        """
        Initialize the Printer object with the specified string.
        Args:
            string (str): A string to be prepended to the printed output.
        """
        self.string = string

    def __call__(self, arg):
        """
        Call method to print information based on the type of the input argument.
        Args:
            arg (str, torch.Tensor, np.ndarray, or any): The input argument to be printed.
        Returns:
            arg (str, torch.Tensor, np.ndarray, or any): The input argument, unchanged.
        """
        if isinstance(arg, str):
            print(self.string, arg)
        elif isinstance(arg, (torch.Tensor, torch.FloatTensor, np.ndarray)):
            print(self.string, "(Shape =", arg.shape, ")")
        else:
            print(self.string, f"({type(arg)})")
        return arg


class SaveImageKeysd:
    """
    A callable class to save specific keys from input data as NIfTI files.
    Args:
        keys (list): List of keys corresponding to the data to be saved.
        output_dir (str): Directory path where the NIfTI files will be saved.
        suffix (str, optional): Optional suffix to be added to the filename of each saved NIfTI file.
    Returns:
        dict: The input data dictionary, unchanged.

    Example:
        >>> saver = SaveImageKeysd(keys=["center_heatmap", "offsets"], output_dir="./output", suffix="processed")
        >>> data = {"center_heatmap": torch.rand(1, 1, 64, 64, 64), "offsets": np.random.rand(1, 3, 64, 64, 64)}
        >>> _ = saver(data)
        The keys "center_heatmap" and "offsets" are saved as NIfTI files in the directory "./output" with the suffix "processed".
    """

    def __init__(self, keys: list, output_dir: str, suffix: str = ""):
        """
        Initialize the SaveImageKeysd object with the specified keys, output directory, and optional suffix.
        Args:
            keys (list): List of keys corresponding to the data to be saved.
            output_dir (str): Directory path where the NIfTI files will be saved.
            suffix (str, optional): Optional suffix to be added to the filename of each saved NIfTI file.
        """
        self.keys = keys
        self.output_dir = output_dir
        self.suffix = suffix

    def __call__(self, data: dict) -> dict:
        """
        Call method to save the specified keys from the input data as NIfTI files.
        Args:
            data (dict): Input data dictionary containing the keys to be saved.
        Returns:
            dict: The input data dictionary, unchanged.
        """
        for key in self.keys:
            image = deepcopy(data[key])

            if key == "center_heatmap":
                image = torch.from_numpy(image) if type(image) == np.ndarray else image
                nms_padding = (3 - 1) // 2
                ctr_hmp = F.max_pool3d(image, kernel_size=3, stride=1, padding=nms_padding)
                ctr_hmp[ctr_hmp != image] = 0
                ctr_hmp[ctr_hmp > 0] = 1
                image = ctr_hmp

            if isinstance(image, (torch.Tensor, MetaTensor)):
                image = image.cpu().numpy()

            image = np.squeeze(image)
            squeeze_dim = 4 if key == "offsets" else 3
            while len(image.shape) > squeeze_dim:
                image = image[0, :]

            if key == "offsets":
                image = image.transpose(1, 2, 3, 0)  # itksnap readable

            if self.suffix != "":
                filename = f"{key}_{self.suffix}.nii.gz"
            else:
                filename = f"{key}.nii.gz"

            filename = os.path.join(self.output_dir, filename)
            nib.save(nib.Nifti1Image(image, np.eye(4)), filename)

        return data


class Printerd:
    """
    A callable class to print information about specific keys in input data.
    Args:
        keys (list): List of keys corresponding to the data to be printed.
        message (str, optional): Optional message to be printed before the key information.
    Returns:
        dict: The input data dictionary, unchanged.

    Example:
        >>> printer = Printerd(keys=["image", "label"], message="Info:")
        >>> data = {"image": np.random.rand(64, 64), "label": np.random.randint(0, 2, (64, 64))}
        >>> _ = printer(data)
        Info: image float64
        Info: label int64
    """

    def __init__(self, keys: list, message: str = ""):
        """
        Initialize the Printerd object with the specified keys and optional message.

        Args:
            keys (list): List of keys corresponding to the data to be printed.
            message (str, optional): Optional message to be printed before the key information.
        """
        self.keys = keys
        self.message = message

    def __call__(self, data: dict) -> dict:
        """
        Call method to print information about the specified keys in the input data.

        Args:
            data (dict): Input data dictionary containing the keys to be printed.

        Returns:
            dict: The input data dictionary, unchanged.
        """
        for key in self.keys:
            image = data[key]
            print(self.message, key, image.dtype)
        return data


class BinarizeInstancesd(MapTransform):
    """
    Binarize the values of specified keys in the input dictionary.
    Args:
        keys (list): List of keys corresponding to the data to be binarized.
        out_key (str, optional): Output key for the binarized data. Defaults to "label".
    Returns:
        dict: The input data dictionary with binarized values.

    Example:
        >>> binarizer = BinarizeInstancesd(keys=["image", "mask"], out_key="binarized")
        >>> data = {"image": np.random.rand(64, 64), "mask": np.random.rand(64, 64)}
        >>> transformed_data = binarizer(data)
    """

    def __init__(self, keys: list, out_key: str = "label"):
        """
        Initializes the BinarizeInstancesd transform with the specified keys and output key.

        Args:
            keys (list): List of keys corresponding to the data to be binarized.
            out_key (str, optional): Output key for the binarized data. Defaults to "label".
        """
        super().__init__(keys)
        self.keys = keys
        self.out_key = out_key

    def __call__(self, data: dict) -> dict:
        """
        Binarizes the values of specified keys in the input dictionary.
        Args:
            data (dict): Input data dictionary containing the keys to be binarized.
        Returns:
            dict: The input data dictionary with binarized values.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            out_key = self.out_key + "_" + key if len(self.keys) > 1 else self.out_key
            assert np.all(np.unique(data[key]) >= 0), "The input data should be non-negative."
            image = deepcopy(data[key])
            image[image > 0] = 1
            d[out_key] = image.astype(np.uint8)
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
