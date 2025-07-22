from typing import Sequence, Optional, Union, Tuple
import numpy as np
import torch
from monai.transforms import MapTransform, Randomizable
from monai.config import KeysCollection
from monai.utils import ensure_tuple_rep

GENERATION_LABELS = [  0,  14,  15,  16,  24,  72,  85, 502, 506, 507, 508, 509, 511,
                     512, 514, 515, 516, 530,   2,   3,   4,   5,   7,   8,  10,  11,
                      12,  13,  17,  18,  25,  26,  28,  30, 136, 137,  41,  42,  43,
                      44,  46,  47,  49,  50,  51,  52,  53,  54,  57,  58,  60,  62,
                     163, 164]

OUTPUT_LABELS = [ 0, 14, 15, 16, 24,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                  0,  0,  2,  3,  4,  5,  7,  8, 10, 11, 12, 13, 17, 18,  2, 26, 28,
                  0,  4,  5, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 41, 58,
                 60,  0, 43, 44]

# LESION_LABELS = list(range(600, 1500))
#LESION_LABELS = list(range(86,250)) + list(range(256, 2000))
LESION_LABELS = list(range(100, 400)) + list(range(1000, 1200))


class RandomLabelsToImaged(MapTransform, Randomizable):
    """
    MONAI version of TorchIO's RandomLabelsToImage:
    Generate a synthetic image from a label map with random Gaussian sampling per label.
    """

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        image_key: str,
        used_labels: Optional[Sequence[int]] = None,
        mean: Optional[Sequence[Union[float, Tuple[float, float]]]] = None,
        std: Optional[Sequence[Union[float, Tuple[float, float]]]] = None,
        default_mean: Tuple[float, float] = (25, 255),
        default_std: Tuple[float, float] = (5, 25),
        discretize: bool = False,
        ignore_background: bool = False,
    ):
        super().__init__(keys)
        self.label_key = label_key
        self.image_key = image_key
        self.used_labels = used_labels
        self.mean = mean
        self.std = std
        self.default_mean = default_mean
        self.default_std = default_std
        self.discretize = discretize
        self.ignore_background = ignore_background

    def randomize(self) -> None:
        self._rand = self.R.uniform()

    def _sample_value(self, value_range: Union[float, Tuple[float, float]]) -> float:
        if isinstance(value_range, (float, int)):
            return float(value_range)
        else:
            return float(self.R.uniform(*value_range))

    def __call__(self, data):
        d = dict(data)
        label_map = d[self.label_key]  # shape: [C, H, W, D] or [C, H, W]

        if isinstance(label_map, np.ndarray):
            label_map = torch.from_numpy(label_map)

        label_map = label_map.long()
        orig_device = label_map.device if isinstance(label_map, torch.Tensor) else torch.device("cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        label_map = label_map.to(device)

        if label_map.ndim > 3:
            if self.discretize:
                label_map = torch.argmax(label_map, dim=0)
            else:
                label_map = label_map[0]  # take first channel

        labels_in_image = torch.unique(label_map)
        if self.used_labels is not None:
            labels_in_image = labels_in_image[
                torch.isin(labels_in_image, torch.tensor(self.used_labels, device=device))]
        if self.ignore_background:
            labels_in_image = labels_in_image[labels_in_image != 0]

        num_labels = labels_in_image.shape[0]
        shape = label_map.shape


        mean_variation = (self.default_mean[1] - self.default_mean[0]) / 5
        std_variation = (self.default_std[1] - self.default_std[0]) / 10

        lesion_mean_1 = self._sample_value(self.default_mean)
        lesion_mean_2 = self._sample_value(
            (lesion_mean_1 - mean_variation, lesion_mean_1 + mean_variation))

        lesion_std_1 = self._sample_value(self.default_std)
        lesion_std_2 = self._sample_value((lesion_std_1 - std_variation, lesion_std_1 + std_variation))

        #lesion_mean = self._sample_value((min(lesion_mean_1, lesion_mean_2), max(lesion_mean_1, lesion_mean_2)))
        #lesion_std = self._sample_value((min(lesion_std_1, lesion_std_2), max(lesion_std_1, lesion_std_2)))
        
        label_chunk_size = 32  # adjustable
        
        while label_chunk_size > 3:
            try:    
                synthetic_image = torch.zeros(shape, dtype=torch.float16, device=device)

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    for start_idx in range(0, num_labels, label_chunk_size):
                        end_idx = min(start_idx + label_chunk_size, num_labels)
                        chunk_labels = labels_in_image[start_idx:end_idx]

                        masks = torch.stack([(label_map == lbl) for lbl in chunk_labels], dim=0).half()  # [B, H, W, D]
                        B = masks.shape[0]

                        means = []
                        stds = []
                        for idx, label in enumerate(chunk_labels):
                            if label.item() in LESION_LABELS:
                                mean = self._sample_value((min(lesion_mean_1, lesion_mean_2), max(lesion_mean_1, lesion_mean_2)))
                                std = self._sample_value((min(lesion_std_1, lesion_std_2), max(lesion_std_1, lesion_std_2)))
                                #mean = lesion_mean
                                #std = lesion_std
                            else:
                                idx_global = (start_idx + idx)
                                mean_range = self.default_mean if self.mean is None else self.mean[idx_global]
                                std_range = self.default_std if self.std is None else self.std[idx_global]
                                mean = self._sample_value(mean_range)
                                std = self._sample_value(std_range)

                            means.append(mean)
                            stds.append(std)

                        means_tensor = torch.tensor(means, device=device, dtype=torch.float16).view(B, 1, 1, 1)
                        stds_tensor = torch.tensor(stds, device=device, dtype=torch.float16).view(B, 1, 1, 1)

                        noise = torch.randn((B, *shape), device=device, dtype=torch.float16) * stds_tensor + means_tensor

                        synthetic_image += (noise * masks).sum(dim=0)

                    break

            except torch.OutOfMemoryError as e:
                for var in ["means_tensor", "stds_tensor", "noise", "synthetic_image", "masks"]:
                    if var in locals():
                        del locals()[var]
                torch.cuda.empty_cache()
                label_chunk_size //= 2

        d[self.image_key] = synthetic_image.float().unsqueeze(0).to(orig_device).numpy()  # Add channel dimension back

        return d


if __name__ == '__main__':
    from monai.transforms import Compose, RandGaussianSmoothd, OneOf, RandAffined, Rand3DElasticd, RandFlipd, RandBiasFieldd, RandGaussianNoised, \
        RandAdjustContrastd, NormalizeIntensityd, ScaleIntensityd
    from conflunet.dataloading.transforms.data_augmentations.motion import RandMotiond
    from conflunet.dataloading.transforms.data_augmentations.simulatelowresolution import RandSimulateLowResolutiond

    # label_path = "/home/mwynen/sub-056_labels.nii.gz"
    label_path = "/home/mwynen/sub-056_labels_instances.nii.gz"
    transform = Compose([
        RandomLabelsToImaged(
            keys=["label"],
            label_key="label",
            image_key="image",
            used_labels=None,  # Assuming labels 1-999 are used
            default_mean=(0.1, 0.9),
            default_std=(0.01, 0.05),
            discretize=False,
            ignore_background=True,
        ),
        # Optionally:
        # RandGaussianSmoothd(keys="image", sigma_x=(0.5, 1.5), prob=0.5),
    ])

    train_transforms = Compose([
        # RandAffined(
        #     prob=1.0,
        #     keys=["image", "label"],
        #     rotate_range=[0, 0, np.deg2rad(20)],
        #     scale_range=[0.4, 0.4, 0.4],  # since scales=(0.6,1.1) => +/- 0.4
        #     translate_range=[10, 10, 10],
        #     padding_mode=('border', 'border'),
        #     mode=('bilinear', 'nearest')
        # ),
        Rand3DElasticd(
            prob=1.0,
            keys=["image", "label"],
            sigma_range=(7.5, 7.5),  # approximate, not exact equivalent
            magnitude_range=(7.5, 7.5),
            translate_range=[10, 10, 10],
            # rotate_range=[0, 0, np.deg2rad(20)],
            # scale_range=[0.4, 0.4, 0.4],
            padding_mode=('border', 'border'),
            mode=('bilinear', 'nearest')
        ),
        RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.74, 1.35)),
        RandFlipd(keys=["image", "label"], prob=1., spatial_axis=(0, 1, 2)),  # Apply flipping along all axes
        RandBiasFieldd(keys=["image"], prob=0.8, coeff_range=(0.1, 0.2), degree=3),
        RandGaussianNoised(keys=["image"], prob=1.0, mean=0.005, std=0.05),
        # RandMotiond(
        #     keys=["image"], degrees=5, translation=1, num_transforms=1,  perturbation=0.3,
        #     image_interpolation='linear', prob=0.02,
        # ),
        RandSimulateLowResolutiond(
            # downsample by NN interpolation and upsample by cubic interpolation
            keys=["image"], downsample_mode="nearest", upsample_mode="trilinear", prob=0.02,
            zoom_range=(0.5, 2.0), align_corners=True
        ),
        # Normalize between 0 and 1
        # NormalizeIntensityd(keys=["image"], channel_wise=True)
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True),
    ])




    import nibabel as nib
    from time import perf_counter
    label_paths = ["/home/mwynen/data/nnUNet/nnUNet_raw/Dataset399_SynConfLUNet/imagesTr/sub-001_0000.nii.gz", # 9 lesions
                   "/home/mwynen/data/nnUNet/nnUNet_raw/Dataset399_SynConfLUNet/imagesTr/sub-005_0000.nii.gz", # 45 lesions
                   "/home/mwynen/data/nnUNet/nnUNet_raw/Dataset399_SynConfLUNet/imagesTr/sub-036_0000.nii.gz"] # 152 lesions
    for label_path in label_paths:
        print(f"Processing {label_path}")
        label_data = nib.load(label_path).get_fdata()
        label_tensor = torch.from_numpy(label_data).unsqueeze(0)  # Add channel dimension
        data = {"label": label_tensor}
        times = []
        for i in range(5):
            start = perf_counter()
            transformed_data = transform(data)
            transformed_data = train_transforms(transformed_data)
            end = perf_counter()
            print(f"Transformation time: {end - start:.4f} seconds")
            times.append(end - start)
            synthetic_image = transformed_data["image"]
            # print(synthetic_image.shape)
            # Save or visualize synthetic_image as needed
            synthetic_image = synthetic_image.numpy()  # Convert to numpy for saving or visualization
            synthetic_image = np.squeeze(synthetic_image)  # Remove channel dimension if needed
            #
            # label = transformed_data["label"]
            # label = label.numpy()
            # label = np.squeeze(label)

            # nib.save(nib.Nifti1Image(synthetic_image, np.eye(4)), label_path.replace('labels.nii.gz', f'synthetic_image_{i}.nii.gz'))
            # nib.save(nib.Nifti1Image(label, np.eye(4)), label_path.replace('labels.nii.gz', f'synthetic_label_{i}.nii.gz'))
            # nib.save(nib.Nifti1Image(synthetic_image, np.eye(4)), label_path.replace('labels_instances.nii.gz', f'synthetic_image_{i}.nii.gz'))
            # nib.save(nib.Nifti1Image(label, np.eye(4)), label_path.replace('labels_instances.nii.gz', f'synthetic_label_{i}.nii.gz'))
            # nib.save(nib.Nifti1Image(synthetic_image, np.eye(4)), label_path.replace('.nii.gz', f'_synthetic_image_{i}.nii.gz'))
            # nib.save(nib.Nifti1Image(label, np.eye(4)), label_path.replace('.nii.gz', f'_synthetic_label_{i}.nii.gz'))
            nib.save(nib.Nifti1Image(synthetic_image, np.eye(4)), f"synthetic_image_{i}.nii.gz")
            break
        print(f"This label took {np.mean(times):.4f} seconds on average to transform")
