import torch
import numpy as np
import nibabel as nib
import os
from os.path import join as pjoin

from monai.data import CacheDataset, DataLoader
from monai.transforms import LoadImaged, AddChanneld, RandCropByPosNegLabeld, RandSpatialCropd, RandFlipd, \
    RandRotate90d, RandAffined, ToTensord, ConcatItemsd
from scipy.ndimage import rotate

from conflunet.utils import BinarizeInstancesd, LesionOffsetTransformd
from data_load import get_train_transforms, get_val_transforms


def create_3d_oval(shape, center, radius):
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_from_center = (
            ((x - center[0]) / radius[0]) ** 2 +
            ((y - center[1]) / radius[1]) ** 2 +
            ((z - center[2]) / radius[2]) ** 2
    )
    mask = dist_from_center <= 1
    return mask


def rotate_3d_oval(ovoid):
    # Calculate the angle for rotation (random angle)
    angle = np.random.uniform(0, 360)

    # Rotate the ovoid using nearest interpolation
    rotated_ovoid = rotate(ovoid.astype(float), angle, order=0, mode='constant', cval=0, reshape=False)

    return rotated_ovoid.astype(bool)


def generate_random_radius(max_radius):
    largest_radius = np.random.randint(5, max_radius + 1)
    smallest_radius = np.random.randint(largest_radius // 2 - 1, largest_radius)
    medium_radius = np.random.randint(smallest_radius, largest_radius)
    radii = np.random.permutation([largest_radius, smallest_radius, medium_radius])
    return tuple(radii)


def generate_random_3d_oval(shape, max_radius, data, seed=0, max_tries=50):
    tries = 0
    while True and tries < max_tries:
        np.random.seed(seed + tries)
        center = tuple(np.random.randint(0, dim) for dim in shape) # Generate random center
        r1, r2, r3 = generate_random_radius(max_radius)
        ovoid = create_3d_oval(shape, center, (r1, r2, r3))
        # Randomly rotate the ovoid
        ovoid = rotate_3d_oval(ovoid)

        if not np.sum(data & ovoid) > 0: # Check if the ovoid intersects with any other ovoid
            return ovoid
        tries += 1 # If the ovoid intersects with another ovoid, try again


def make_image(size=(128, 128, 128), directory=".temp", seed=0):
    # Define the shape of the 3D numpy matrix
    matrix_shape = (128, 128, 128)

    # Define the maximal size of the ovoid shapes
    max_size = 20

    # Create the 3D matrix with ovoid shapes
    semantic_seg = np.zeros(matrix_shape, dtype=bool)
    instance_seg = np.zeros(matrix_shape, dtype=np.uint32)
    for i in range(10):  # Number of ovoid shapes to create
        ovoid = generate_random_3d_oval(matrix_shape, max_size, semantic_seg, seed=seed+(i*50), max_tries=50)
        semantic_seg |= ovoid
        instance_seg[ovoid] = i + 1

    # Save the result to a .nii.gz file
    sid = str(seed).zfill(3)
    img = (semantic_seg * 100) + 10
    img = nib.Nifti1Image(img.astype(np.float32), affine=np.eye(4))
    nib.save(img, pjoin(directory, f"sub-{sid}_image.nii.gz"))
    img = nib.Nifti1Image(semantic_seg.astype(np.uint8), affine=np.eye(4))
    nib.save(img, pjoin(directory, f"sub-{sid}_mask-classes.nii.gz"))
    img = nib.Nifti1Image(instance_seg, affine=np.eye(4))
    nib.save(img, pjoin(directory, f"sub-{sid}_mask-instances.nii.gz"))


def generate_synthetic_data(overwrite=False):
    DATA_ROOT_DIR = os.getenv('DATA_ROOT_DIR')
    if DATA_ROOT_DIR is None:
        DATA_ROOT_DIR = "."
    if os.path.exists(pjoin(DATA_ROOT_DIR, ".temp", "sub-000_image.nii.gz")) and not overwrite:
        return
    os.makedirs(pjoin(DATA_ROOT_DIR, ".temp"), exist_ok=True)
    for i in range(10):
        make_image(seed=i)


def synthetic_dataset():
    DATA_ROOT_DIR = os.getenv('DATA_ROOT_DIR', '.')
    files = {
        "image": [pjoin(DATA_ROOT_DIR, ".temp", file) for file in os.listdir(pjoin(DATA_ROOT_DIR, ".temp")) if "image" in file],
        "instance_mask": [pjoin(DATA_ROOT_DIR, ".temp", file) for file in os.listdir(pjoin(DATA_ROOT_DIR, ".temp")) if "mask-instances" in file]
    }
    # transform files in a list of dicts
    files = [
        {"image": img, "instance_mask": instance_mask}
        for img, instance_mask in zip(sorted(files["image"]), sorted(files["instance_mask"]))
    ]

    I = ["image"]
    masks = ["instance_mask"]

    transforms = [
        LoadImaged(keys=I + masks),
        AddChanneld(keys=I + masks),
        # # Lambdad(keys=non_label_masks, func=lambda x: x.astype(np.uint), allow_missing_keys=True),
        # NormalizeIntensityd(keys=non_quantitative_images, nonzero=True),
        # RandShiftIntensityd(keys=non_quantitative_images, offsets=0.1, prob=1.0),
        # RandScaleIntensityd(keys=non_quantitative_images, factors=0.1, prob=1.0),
        BinarizeInstancesd(keys=masks),
        RandCropByPosNegLabeld(keys=I + ["instance_mask", "label"],
                               label_key="label", image_key=I[0],
                               spatial_size=(128, 128, 128), num_samples=32,
                               pos=4, neg=1),
        # RandSpatialCropd(keys=I + ["instance_mask", "label"],
        #                  roi_size=(96, 96, 96),
        #                  random_center=True, random_size=False),
        # RandFlipd(keys=I + ["instance_mask", "label"], prob=0.5, spatial_axis=(0, 1, 2)),
        # RandRotate90d(keys=I + ["instance_mask", "label"], prob=0.5, spatial_axes=(0, 1)),
        # RandRotate90d(keys=I + ["instance_mask", "label"], prob=0.5, spatial_axes=(1, 2)),
        # RandRotate90d(keys=I + ["instance_mask", "label"], prob=0.5, spatial_axes=(0, 2)),
        # RandAffined(keys=I + ["instance_mask", "label"],
        #             mode=tuple(['bilinear'] * len(I)) + ('nearest', 'nearest'),
        #             prob=1.0, spatial_size=(96, 96, 96),
        #             rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
        #             scale_range=(0.1, 0.1, 0.1), padding_mode='border'),
        # LesionOffsetTransformd(keys="instance_mask"),
        # ToTensord(keys=I + other_keys),
        # ConcatItemsd(keys=I, name="image", dim=0)
    ]

    ds = CacheDataset(data=files, transform=transforms, cache_rate=0, num_workers=1)
    return ds
    # return DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)


def visualize_synthetic_data(save_dir=".temp"):
    import matplotlib.pyplot as plt
    from monai.visualize import plot_2d_or_3d_image
    ds = synthetic_dataset()
    print("Length of dataset:", len(ds))
    for i in range(len(ds)):  # Loop through dataset indices
        this = ds[i]
        img, label, instance_mask = this[0]["image"], this[0]["label"], this[0]["instance_mask"]

        # Save the images
        img = img[0].numpy().squeeze()
        label = label[0].numpy().squeeze()
        instance_mask = instance_mask[0].numpy().squeeze()

        # Save as nifti in the save_dir
        img = nib.Nifti1Image(img, np.eye(4))
        label = nib.Nifti1Image(label, np.eye(4))
        instance_mask = nib.Nifti1Image(instance_mask, np.eye(4))
        nib.save(img, pjoin(save_dir, f"sub-{str(i).zfill(3)}_image_DATALOAD.nii.gz"))
        nib.save(label, pjoin(save_dir, f"sub-{str(i).zfill(3)}_label_DATALOAD.nii.gz"))
        nib.save(instance_mask, pjoin(save_dir, f"sub-{str(i).zfill(3)}_instance_mask_DATALOAD.nii.gz"))




if __name__ == "__main__":
    generate_synthetic_data(overwrite=True)
    visualize_synthetic_data()
    # from monai.transforms import LoadImaged, AddChanneld, RandCropByPosNegLabeld, RandSpatialCropd
    # from utils import BinarizeInstancesd
    #
    # DATA_ROOT_DIR = os.getenv('DATA_ROOT_DIR', '.')
    # files = {
    #     "image": [pjoin(DATA_ROOT_DIR, ".temp", file) for file in os.listdir(pjoin(DATA_ROOT_DIR, ".temp")) if
    #               "image" in file],
    #     "instance_mask": [pjoin(DATA_ROOT_DIR, ".temp", file) for file in os.listdir(pjoin(DATA_ROOT_DIR, ".temp")) if
    #                       "mask-instances" in file]
    # }
    # # transform files in a list of dicts
    # files = [
    #     {"image": img, "instance_mask": instance_mask}
    #     for img, instance_mask in zip(sorted(files["image"]), sorted(files["instance_mask"]))
    # ]
    # t1 = LoadImaged(keys=["image", "instance_mask"])
    # t2 = AddChanneld(keys=["image", "instance_mask"])
    # t3 = BinarizeInstancesd(keys=["instance_mask"])
    # t4 = RandCropByPosNegLabeld(keys=["image", "instance_mask", "label"],
    #                            label_key="label", image_key="image",
    #                            spatial_size=(128, 128, 128), num_samples=32,
    #                            pos=4, neg=1)
    # t5 = RandSpatialCropd(keys= ["image", "instance_mask", "label"],
    #                       roi_size=(96, 96, 96),
    #                       random_center=True, random_size=False)
    #
    # file = files[0]
    # file = t1(file)
    # file = t2(file)
    # file = t3(file)
    # file = t4(file)
    # file = t5(file)
    # print()