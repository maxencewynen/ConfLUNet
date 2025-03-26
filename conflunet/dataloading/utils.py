import warnings
import numpy as np
from typing import Callable, Tuple, Sequence, Mapping, Hashable, Any, Union

from monai.transforms import (
    ToTensord,
    RandFlipd,
    RandScaleIntensityd,
    FgBgToIndicesd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Compose,
    RandAffined,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
)

from conflunet.dataloading.transforms.data_augmentations.scaleintensityfixedmean import RandScaleIntensityFixedMeand
from conflunet.dataloading.transforms.data_augmentations.adjustcontrast import RandAdjustContrastd
from conflunet.dataloading.transforms.data_augmentations.simulatelowresolution import RandSimulateLowResolutiond
from conflunet.dataloading.transforms.loading import CustomLoadNPZInstanced, LesionOffsetTransformd, \
    RemoveSmallInstancesTransform
from conflunet.dataloading.transforms.utils import *


def get_nnunet_spatial_transforms(image_key: str = "img",
                                  seg_keys: Tuple = ("seg", "instance_seg"),
                                  spatial_size: Tuple = ()) -> Sequence[Callable]:
    keys = [image_key] + seg_keys
    interpolation = ("bilinear",) + ("nearest",) * len(seg_keys)
    
    transforms = [
        RandAffined(
            keys=keys,
            mode=interpolation,
            prob=0.2,
            spatial_size=spatial_size,
            scale_range=((-0.3, 0.4),) * 3,
            padding_mode='zeros'
            ),
        RandAffined(
            keys=keys,
            mode=interpolation,
            prob=0.2,
            spatial_size=spatial_size,
            rotate_range=(0.52, ) * 3,
            padding_mode='zeros'
            ),
        RandAffined(
            keys=keys,
            mode=interpolation,
            prob=0.2,
            spatial_size=spatial_size,
            translate_range=(10, ) * 3,
            padding_mode='zeros'
            ),
        ]
    return transforms


def get_nnunet_augmentations(image_key: str = "img", seg_keys: Sequence[str] = ("seg", "instance_seg"))\
        -> Sequence[Callable]:
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
            keys=image_key, gamma=(0.7, 1.5), prob=0.1,
            invert_image=True, retain_stats=True
        )
    )

    transforms.append(
        RandAdjustContrastd(
            keys=image_key, gamma=(0.7, 1.5), prob=.3,
            invert_image=False, retain_stats=True
        )
    )

    # Random mirroring
    transforms.append(
        RandFlipd(keys=keys, spatial_axis=(0, 1, 2), prob=0.5,),
    )

    return transforms


def get_train_transforms(seed: Union[int, None] = None,
                         patch_size: Tuple = (128, 128, 128),
                         remove_small_instances: bool = False,
                         voxel_size: Tuple = (1, 1, 1),
                         minimum_instance_size: int = 14,
                         minimum_size_along_axis: int = 3,
                         get_small_instances=False,
                         get_confluent_instances=False,
                         ) -> Compose:
    additional_keys = []
    if get_small_instances or get_confluent_instances:
        additional_keys.append('weights')

    transform_list = [
        CustomLoadNPZInstanced(keys=['data'], get_small_instances=get_small_instances, get_confluent_instances=get_confluent_instances),
        FgBgToIndicesd(keys=['seg']),
        # # Crop random fixed sized regions with the center being a foreground or background voxel
        # # based on the Pos Neg Ratio.
        RandCropByPosNegLabeld(keys=['img', 'seg', 'instance_seg', 'brainmask'] + additional_keys,
                               label_key="seg",
                               fg_indices_key="seg_fg_indices",
                               bg_indices_key="seg_bg_indices",
                               spatial_size=patch_size, num_samples=1,
                               pos=1, neg=1, allow_smaller=True),
        *get_nnunet_spatial_transforms(image_key="img", seg_keys=["seg", "instance_seg", 'brainmask'] + additional_keys,
            spatial_size=patch_size),
        # RandSpatialCropd(keys=['img', 'seg', 'instance_seg', 'brainmask'],
        #                  roi_size=patch_size,
        #                  random_center=False, random_size=False),
        *get_nnunet_augmentations(image_key="img", seg_keys=["seg", "instance_seg", 'brainmask'] + additional_keys),
        LesionOffsetTransformd(keys="instance_seg"),
        ToTensord(keys=['img', 'seg', 'offsets', 'center_heatmap', 'brainmask'] + additional_keys),
        DeleteKeysd(keys=['properties']),
    ]
    if remove_small_instances:
        transform_list.insert(1, RemoveSmallInstancesTransform(
            keys=["instance_seg", "seg"], instance_seg_key="instance_seg", voxel_size=voxel_size,
            minimum_instance_size=minimum_instance_size, minimum_size_along_axis=minimum_size_along_axis)
                              )
    transform = Compose(transform_list)

    if seed is not None:
        transform.set_random_state(seed=seed)
    return transform


def get_val_transforms(seed: Union[int, None] = None,
                       patch_size: Tuple = (128, 128, 128),
                       remove_small_instances: bool = False,
                       voxel_size: Tuple = (1, 1, 1),
                       minimum_instance_size: int = 14,
                       minimum_size_along_axis: int = 3
                       ) -> Compose:

    transform_list = [
        CustomLoadNPZInstanced(keys=['data']),
        FgBgToIndicesd(keys=['seg']),
        # Crop random fixed sized regions with the center being a foreground or background voxel
        # based on the Pos Neg Ratio.
        RandCropByPosNegLabeld(keys=['img', 'seg', 'instance_seg', 'brainmask'],
                               label_key="seg",
                               fg_indices_key="seg_fg_indices",
                               bg_indices_key="seg_bg_indices",
                               spatial_size=patch_size, num_samples=1,
                               pos=1, neg=1, allow_smaller=True),
        LesionOffsetTransformd(keys="instance_seg"),
        ToTensord(keys=['img', 'seg', 'offsets', 'center_heatmap', 'brainmask']),
        DeleteKeysd(keys=['properties']),
    ]
    if remove_small_instances:
        transform_list.insert(1, RemoveSmallInstancesTransform(
            keys=["instance_seg", "seg"], instance_seg_key="instance_seg", voxel_size=voxel_size,
            minimum_instance_size=minimum_instance_size, minimum_size_along_axis=minimum_size_along_axis)
                              )
    transform = Compose(transform_list)

    if seed is not None:
        transform.set_random_state(seed=seed)
    return transform


def get_test_transforms(test: bool = True, voxel_size: Tuple = (1, 1, 1),
                        minimum_instance_size: int = 14, minimum_size_along_axis: int = 3) -> Compose:
    transform_list = [
        CustomLoadNPZInstanced(keys=['data'], test=test),
        ToTensord(keys=['img', 'brainmask'], allow_missing_keys=True),
        DeleteKeysd(keys=['properties']),
    ]
    if not test:
        transform_list.insert(1, RemoveSmallInstancesTransform(
            keys=["instance_seg", "seg"], instance_seg_key="instance_seg", voxel_size=voxel_size,
            minimum_instance_size=minimum_instance_size, minimum_size_along_axis=minimum_size_along_axis
        ))
    transform = Compose(transform_list)
    return transform
