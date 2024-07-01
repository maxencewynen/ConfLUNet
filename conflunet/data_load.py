"""
adapted from https://github.com/Shifts-Project/shifts/tree/main/mswml
"""
import numpy as np
import os
from os.path import join as pjoin
from glob import glob
import re
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AddChanneld, Compose, LoadImaged, RandCropByPosNegLabeld,
    Spacingd, ToTensord, NormalizeIntensityd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, RandAffined, RandSpatialCropd,
    RandScaleIntensityd, RandSpatialCropSamplesd, ConcatItemsd, Lambdad,
    MaskIntensityd)
from utils import Printer, Printerd, LesionOffsetTransformd, SaveImageKeysd, BinarizeInstancesd

DISCARDED_SUBJECTS = []
QUANTITATIVE_SEQUENCES = ["T1map"]


def get_train_transforms(I=['FLAIR'], apply_mask=None):
    """ Get transforms for training on FLAIR images and ground truth:
    - Loads 3D images from Nifti file
    - Adds channel dimention
    - Normalises intensity
    - Applies augmentations
    - Crops out 32 patches of shape [96, 96, 96] that contain lesions
    - Converts to torch.Tensor()
    """

    masks = ["instance_mask"]  # , "brain_mask"]
    non_label_masks = []
    if apply_mask:
        masks += [apply_mask]
        non_label_masks += [apply_mask]

    other_keys = ["label", "center_heatmap", "offsets"]

    non_quantitative_images = [i for i in I if i not in QUANTITATIVE_SEQUENCES]

    transform_list = [
        LoadImaged(keys=I + masks),
        AddChanneld(keys=I + masks),
        # Lambdad(keys=non_label_masks, func=lambda x: x.astype(np.uint), allow_missing_keys=True),
        NormalizeIntensityd(keys=non_quantitative_images, nonzero=True),
        RandShiftIntensityd(keys=non_quantitative_images, offsets=0.1, prob=1.0),
        RandScaleIntensityd(keys=non_quantitative_images, factors=0.1, prob=1.0),
        BinarizeInstancesd(keys=masks),
    ]
    if apply_mask:
        transform_list += [MaskIntensityd(keys=I + masks, mask_key=apply_mask)]

    transform_list += [
        RandCropByPosNegLabeld(keys=I + ["instance_mask", "label"],
                               label_key="label", image_key=I[0],
                               spatial_size=(128, 128, 128), num_samples=32,
                               pos=4, neg=1),
        RandSpatialCropd(keys=I + ["instance_mask", "label"],
                         roi_size=(96, 96, 96),
                         random_center=True, random_size=False),
        RandFlipd(keys=I + ["instance_mask", "label"], prob=0.5, spatial_axis=(0, 1, 2)),
        RandRotate90d(keys=I + ["instance_mask", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandRotate90d(keys=I + ["instance_mask", "label"], prob=0.5, spatial_axes=(1, 2)),
        RandRotate90d(keys=I + ["instance_mask", "label"], prob=0.5, spatial_axes=(0, 2)),
        RandAffined(keys=I + ["instance_mask", "label"],
                    mode=tuple(['bilinear'] * len(I)) + ('nearest', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                    scale_range=(0.1, 0.1, 0.1), padding_mode='border'),
        LesionOffsetTransformd(keys="instance_mask"),
        ToTensord(keys=I + other_keys),
        ConcatItemsd(keys=I, name="image", dim=0)
    ]
    # transform.set_random_state(seed=seed)

    return Compose(transform_list)


def get_val_transforms(I=['FLAIR'], bm=False, apply_mask=None):
    """ Get transforms for testing on FLAIR images and ground truth:
    - Loads 3D images and masks from Nifti file
    - Adds channel dimention
    - Applies intensity normalisation to scans
    - Converts to torch.Tensor()
    """
    other_keys = ["instance_mask", "brain_mask"] if bm else ["instance_mask"]
    other_keys = other_keys + [apply_mask] if apply_mask else other_keys

    non_quantitative_images = [i for i in I if i not in QUANTITATIVE_SEQUENCES]

    transforms = [
        LoadImaged(keys=I + other_keys),
        AddChanneld(keys=I + other_keys),
        # Lambdad(keys=["label"], func=lambda x: (x>0).astype(int) ),
    ]
    transforms = transforms + [Lambdad(keys=["brain_mask"], func=lambda x: x.astype(np.uint8))] if bm else transforms
    transforms = transforms + [MaskIntensityd(keys=I, mask_key=apply_mask)] if apply_mask else transforms

    transforms += [
        Lambdad(keys=["brain_mask"], func=lambda x: x.astype(np.uint8)),
        NormalizeIntensityd(keys=non_quantitative_images, nonzero=True),
        LesionOffsetTransformd(keys="instance_mask", remove_small_lesions=True),
        ToTensord(keys=I + other_keys + ["label", "center_heatmap", "offsets"]),
        ConcatItemsd(keys=I, name="image", dim=0)
    ]
    return Compose(transforms)


def get_test_transforms(I=['FLAIR'], bm=False, apply_mask=None):
    """ Get transforms for testing on FLAIR images and ground truth:
    - Loads 3D images and masks from Nifti file
    - Adds channel dimention
    - Applies intensity normalisation to scans
    - Converts to torch.Tensor()
    """
    other_keys = ["brain_mask"] if bm else []
    other_keys = other_keys + [apply_mask] if apply_mask else other_keys

    non_quantitative_images = [i for i in I if i not in QUANTITATIVE_SEQUENCES]

    transforms = [
        LoadImaged(keys=I + other_keys),
        AddChanneld(keys=I + other_keys),
        # Lambdad(keys=["label"], func=lambda x: (x>0).astype(int) ),
    ]
    transforms = transforms + [Lambdad(keys=["brain_mask"], func=lambda x: x.astype(np.uint8))] if bm else transforms
    transforms = transforms + [MaskIntensityd(keys=I, mask_key=apply_mask)] if apply_mask else transforms

    transforms += [
        Lambdad(keys=["brain_mask"], func=lambda x: x.astype(np.uint8)),
        NormalizeIntensityd(keys=non_quantitative_images, nonzero=True),
        ToTensord(keys=I + other_keys),
        ConcatItemsd(keys=I, name="image", dim=0)
    ]
    return Compose(transforms)


def get_train_dataloader(data_dir, num_workers, cache_rate=0.1, seed=1, I=['FLAIR'], apply_mask=None):
    """
    Get dataloader for training
    Args:
      data_dir: `str`, path to data directory (should contain img/ and labels/).
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      I: `list`, list of modalities to include in the data loader.
    Returns:
      monai.data.DataLoader() class object.
    """
    assert os.path.exists(data_dir), f"data_dir path does not exist ({data_dir})"
    assert apply_mask is None or type(apply_mask) == str
    traindir = "train"
    img_dir = pjoin(data_dir, traindir, "images")
    lbl_dir = pjoin(data_dir, traindir, "labels")
    bm_path = pjoin(data_dir, "train", "brainmasks")
    mask_path = None if not apply_mask else pjoin(data_dir, "train", apply_mask)

    # Collect all modality images sorted
    all_modality_images = {}
    all_modality_images = {
        i: [
            pjoin(img_dir, s)
            for s in sorted(list(os.listdir(img_dir)))
            if s.endswith(i + ".nii.gz") and not any(subj in s for subj in DISCARDED_SUBJECTS)
        ]
        for i in I
    }
    for modality in I:
        for j in range(len([j for j in I if j == modality])):
            if j == 0: continue
            all_modality_images[modality + str(j)] = all_modality_images[modality]

    # Check all modalities have same length
    assert all(len(x) == len(all_modality_images[I[0]]) for x in
               all_modality_images.values()), "All modalities must have the same number of images"

    # Collect all corresponding ground truths
    maskname = "mask-instances"
    segs = [pjoin(lbl_dir, f) for f in sorted(list(os.listdir(lbl_dir))) if f.endswith(maskname + ".nii.gz")]

    assert len(all_modality_images[I[0]]) == len(
        segs), "Number of multi-modal images and ground truths must be the same"

    files = []

    bms = [pjoin(bm_path, f) for f in sorted(list(os.listdir(bm_path))) if f.endswith("brainmask.nii.gz")]
    if not apply_mask:
        assert len(all_modality_images[I[0]]) == len(segs) == len(
            bms), f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs), len(bms)]}"

        for i in range(len(segs)):
            file_dict = {"instance_mask": segs[i], "brain_mask": bms[i]}
            for modality in all_modality_images.keys():  # in I:
                file_dict[modality] = all_modality_images[modality][i]
            files.append(file_dict)

    else:
        masks = [pjoin(mask_path, f) for f in sorted(list(os.listdir(mask_path))) if f.endswith(".nii.gz")]
        assert len(all_modality_images[I[0]]) == len(segs) == len(bms) == len(masks), \
            f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs), len(bms), len(masks)]}"

        for i in range(len(segs)):
            file_dict = {"instance_mask": segs[i], "brain_mask": bms[i], apply_mask: masks[i]}
            for modality in all_modality_images.keys():  # in I:
                file_dict[modality] = all_modality_images[modality][i]
            files.append(file_dict)

    print("Number of training files:", len(files))
    train_transforms = get_train_transforms(list(all_modality_images.keys()), apply_mask=apply_mask)  # I

    for f in files:
        f['subject'] = os.path.basename(f["instance_mask"])[:7]

    ds = CacheDataset(data=files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=True, num_workers=num_workers)


def get_val_dataloader(data_dir, num_workers, cache_rate=0.1, I=['FLAIR'], test=False, apply_mask=None):
    """
    Get dataloader for validation and testing. Either with or without brain masks.

    Args:
      data_dir: `str`, path to data directory (should contain img/ and labels/).
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      bm_path:   `None|str`. If `str`, then defines path to directory with
                 brain masks. If `None`, dataloader does not return brain masks.
      I: `list`, list of I to include in the data loader.
    Returns:
      monai.data.DataLoader() class object.
    """

    assert os.path.exists(data_dir), f"data_dir path does not exist ({data_dir})"
    assert apply_mask is None or type(apply_mask) == str
    img_dir = pjoin(data_dir, "val", "images") if not test else pjoin(data_dir, "test", "images")
    lbl_dir = pjoin(data_dir, "val", "labels") if not test else pjoin(data_dir, "test", "labels")
    bm_path = pjoin(data_dir, "val", "brainmasks") if not test else pjoin(data_dir, "test", "brainmasks")
    if not apply_mask:
        mask_path = None
    else:
        mask_path = pjoin(data_dir, "val", apply_mask) if not test else pjoin(data_dir, "test", apply_mask)

    # Collect all modality images sorted
    all_modality_images = {
        i: [
            pjoin(img_dir, s)
            for s in sorted(list(os.listdir(img_dir)))
            if s.endswith(i + ".nii.gz") and not any(subj in s for subj in DISCARDED_SUBJECTS)
        ]
        for i in I
    }
    for modality in I:
        for j in range(len([j for j in I if j == modality])):
            if j == 0: continue
            all_modality_images[modality + str(j)] = all_modality_images[modality]

    # Check all modalities have same length
    assert all(len(x) == len(all_modality_images[I[0]]) for x in
               all_modality_images.values()), "All modalities must have the same number of images"

    # Collect all corresponding ground truths
    maskname = "mask-instances"
    segs = [pjoin(lbl_dir, f) for f in sorted(list(os.listdir(lbl_dir))) if f.endswith(maskname + ".nii.gz")]

    assert len(all_modality_images[I[0]]) == len(
        segs), "Number of multi-modal images and ground truths must be the same"

    files = []
    if bm_path is not None:
        bms = [pjoin(bm_path, f) for f in sorted(list(os.listdir(bm_path))) if f.endswith("brainmask.nii.gz")]
        if not apply_mask:
            assert len(all_modality_images[I[0]]) == len(segs) == len(
                bms), f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs), len(bms)]}"

            for i in range(len(segs)):
                file_dict = {"instance_mask": segs[i], "brain_mask": bms[i]}
                for modality in all_modality_images.keys():  # in I:
                    file_dict[modality] = all_modality_images[modality][i]
                files.append(file_dict)

        else:
            masks = [pjoin(mask_path, f) for f in sorted(list(os.listdir(mask_path))) if f.endswith(".nii.gz")]
            assert len(all_modality_images[I[0]]) == len(segs) == len(bms) == len(masks), \
                f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs), len(bms), len(masks)]}"

            for i in range(len(segs)):
                file_dict = {"instance_mask": segs[i], "brain_mask": bms[i], apply_mask: masks[i]}
                for modality in all_modality_images.keys():  # in I:
                    file_dict[modality] = all_modality_images[modality][i]
                files.append(file_dict)

        val_transforms = get_val_transforms(list(all_modality_images.keys()), bm=True, apply_mask=apply_mask)

    else:
        if not apply_mask:
            assert len(all_modality_images[I[0]]) == len(
                segs), f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs)]}"
            for i in range(len(segs)):
                file_dict = {"instance_mask": segs[i]}
                for modality in all_modality_images.keys():  # in I:
                    file_dict[modality] = all_modality_images[modality][i]
                files.append(file_dict)
        else:
            bms = [pjoin(bm_path, f) for f in sorted(list(os.listdir(bm_path))) if f.endswith("brainmask.nii.gz")]
            masks = [pjoin(mask_path, f) for f in sorted(list(os.listdir(mask_path))) if f.endswith(".nii.gz")]
            assert len(all_modality_images[I[0]]) == len(segs) == len(bms) == len(masks), \
                f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs), len(bms), len(masks)]}"

            for i in range(len(segs)):
                file_dict = {"instance_mask": segs[i], "brain_mask": bms[i], apply_mask: masks[i]}
                for modality in all_modality_images.keys():  # in I:
                    file_dict[modality] = all_modality_images[modality][i]
                files.append(file_dict)
        val_transforms = get_val_transforms(list(all_modality_images.keys()), apply_mask=apply_mask)

    if test:
        print("Number of test files:", len(files))
    else:
        print("Number of validation files:", len(files))
    for f in files:
        f['subject'] = os.path.basename(f["instance_mask"])[:7]

    ds = CacheDataset(data=files, transform=val_transforms, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=True, num_workers=num_workers)


def get_test_dataloader(data_dir, num_workers, cache_rate=0.1, I=['FLAIR'], apply_mask=None):
    """
    Get dataloader for testing. Either with or without brain masks.

    Args:
      data_dir: `str`, path to data directory (should contain img/ and labels/).
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      bm_path:   `None|str`. If `str`, then defines path to directory with
                 brain masks. If `None`, dataloader does not return brain masks.
      I: `list`, list of I to include in the data loader.
    Returns:
      monai.data.DataLoader() class object.
    """
    assert os.path.exists(data_dir), f"data_dir path does not exist ({data_dir})"
    assert apply_mask is None or type(apply_mask) == str
    img_dir = pjoin(data_dir, "test", "images")
    bm_path = pjoin(data_dir, "test", "brainmasks")
    if not apply_mask:
        mask_path = None
    else:
        mask_path = pjoin(data_dir, "test", apply_mask)

    # Collect all modality images sorted
    all_modality_images = {
        i: [
            pjoin(img_dir, s)
            for s in sorted(list(os.listdir(img_dir)))
            if s.endswith(i + ".nii.gz")
        ]
        for i in I
    }
    for modality in I:
        for j in range(len([j for j in I if j == modality])):
            if j == 0: continue
            all_modality_images[modality + str(j)] = all_modality_images[modality]

    # Check all modalities have same length
    assert all(len(x) == len(all_modality_images[I[0]]) for x in
               all_modality_images.values()), "All modalities must have the same number of images"

    files = []

    if bm_path is not None:
        bms = [pjoin(bm_path, f) for f in sorted(list(os.listdir(bm_path))) if f.endswith("brainmask.nii.gz")]
        if not apply_mask:
            assert len(all_modality_images[I[0]]) == len(bms), \
                f"Some files must be missing: {[len(all_modality_images[I[0]]), len(bms)]}"

            for i in range(len(all_modality_images[I[0]])):
                file_dict = {"brain_mask": bms[i]}
                for modality in all_modality_images.keys():  # in I:
                    file_dict[modality] = all_modality_images[modality][i]
                files.append(file_dict)

        else:
            masks = [pjoin(mask_path, f) for f in sorted(list(os.listdir(mask_path))) if f.endswith(".nii.gz")]
            assert len(all_modality_images[I[0]]) == len(bms) == len(masks), \
                f"Some files must be missing: {[len(all_modality_images[I[0]]), len(bms), len(masks)]}"

            for i in range(len(all_modality_images[I[0]])):
                file_dict = {"brain_mask": bms[i], apply_mask: masks[i]}
                for modality in all_modality_images.keys():  # in I:
                    file_dict[modality] = all_modality_images[modality][i]
                files.append(file_dict)

        test_transforms = get_test_transforms(list(all_modality_images.keys()), bm=True, apply_mask=apply_mask)

    else:
        if not apply_mask:
            for i in range(len(all_modality_images[I[0]])):
                file_dict = {}
                for modality in all_modality_images.keys():  # in I:
                    file_dict[modality] = all_modality_images[modality][i]
                files.append(file_dict)
        else:
            bms = [pjoin(bm_path, f) for f in sorted(list(os.listdir(bm_path))) if f.endswith("brainmask.nii.gz")]
            masks = [pjoin(mask_path, f) for f in sorted(list(os.listdir(mask_path))) if f.endswith(".nii.gz")]
            assert len(all_modality_images[I[0]]) == len(bms) == len(masks), \
                f"Some files must be missing: {[len(all_modality_images[I[0]]), len(bms), len(masks)]}"

            for i in range(len(all_modality_images[I[0]])):
                file_dict = {"brain_mask": bms[i], apply_mask: masks[i]}
                for modality in all_modality_images.keys():  # in I:
                    file_dict[modality] = all_modality_images[modality][i]
                files.append(file_dict)
        test_transforms = get_test_transforms(list(all_modality_images.keys()), apply_mask=apply_mask)

    print("Number of test files:", len(files))
    for f in files:
        f['subject'] = os.path.basename(f[all_modality_images[I[0]][0]])[:7]

    ds = CacheDataset(data=files, transform=test_transforms, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)

if __name__ == "__main__":
    pass