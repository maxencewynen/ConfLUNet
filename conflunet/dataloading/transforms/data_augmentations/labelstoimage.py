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
LESION_LABELS = list(range(86,250)) + list(range(256, 2000))

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

        if label_map.dtype != torch.long:
            label_map = label_map.long()

        if self.discretize and label_map.ndim > 3:
            label_map = torch.argmax(label_map, dim=0, keepdim=True)

        if label_map.ndim == 4:
            labels_in_image = label_map.unique().long().tolist()
        else:
            labels_in_image = torch.unique(label_map).tolist()

        if self.used_labels is not None:
            labels_in_image = [l for l in labels_in_image if l in self.used_labels]

        synthetic_image = torch.zeros_like(label_map[0], dtype=torch.float32)

        # sample two values from default mean that are relatively close to each other
        mean_variation = (self.default_mean[1] - self.default_mean[0]) / 5  # Adjust variation to ensure closeness
        lesion_mean_1 = self._sample_value(self.default_mean)
        lesion_mean_2 = self._sample_value((lesion_mean_1 - mean_variation, lesion_mean_1 + mean_variation))
        lesion_mean = (min(lesion_mean_1, lesion_mean_2), max(lesion_mean_1, lesion_mean_2))
        # sample two values from default std that are relatively close to each other
        std_variation = (self.default_std[1] - self.default_std[0]) / 10  # Adjust variation to ensure closeness
        lesion_std_1 = self._sample_value(self.default_std)
        lesion_std_2 = self._sample_value((lesion_std_1 - std_variation, lesion_std_1 + std_variation))
        lesion_std = (min(lesion_std_1, lesion_std_2), max(lesion_std_1, lesion_std_2))
        # print(f"Lesion mean: {lesion_mean}, Lesion std: {lesion_std}")

        for idx, label in enumerate(labels_in_image):
            if label == 0 and self.ignore_background:
                continue

            if label in LESION_LABELS:
                mean = self._sample_value(lesion_mean)
                std = self._sample_value(lesion_std)
                # print(f"Label {label} : using mean: {mean}, std: {std}")

            else:
                mean_range = self.default_mean if self.mean is None else self.mean[idx]
                std_range = self.default_std if self.std is None else self.std[idx]

                mean = self._sample_value(mean_range)
                std = self._sample_value(std_range)

            mask = (label_map == label)
            noise = torch.randn_like(synthetic_image) * std + mean
            synthetic_image = torch.where(mask, noise, synthetic_image)

        d[self.image_key] = synthetic_image

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
        RandGaussianSmoothd(keys="image", sigma_x=(0.5, 1.5), prob=0.5),
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
        # RandFlipd(keys=["image", "label"], prob=1., spatial_axis=(0, 1, 2)),  # Apply flipping along all axes
        RandBiasFieldd(keys=["image"], prob=0.8, coeff_range=(0.1, 0.2), degree=3),
        RandGaussianNoised(keys=["image"], prob=1.0, mean=0.005, std=0.05),
        # RandMotiond(
        #     keys=["image"], degrees=5, translation=1, num_transforms=1,  perturbation=0.3,
        #     image_interpolation='linear', prob=0.02,
        # ),
        # RandSimulateLowResolutiond(
        #     # downsample by NN interpolation and upsample by cubic interpolation
        #     keys=["image"], downsample_mode="nearest", upsample_mode="trilinear", prob=0.02,
        #     zoom_range=(0.5, 2.0), align_corners=True
        # ),
        # Normalize between 0 and 1
        # NormalizeIntensityd(keys=["image"], channel_wise=True)
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0, channel_wise=True),
    ])




    import nibabel as nib

    label_paths = ["/home/mwynen/synthetic_images/label_lesion_HC_inst_sub-001_0.nii.gz",
                   "/home/mwynen/synthetic_images/label_lesion_HC_inst_sub-162_0.nii.gz",
                   "/home/mwynen/synthetic_images/label_lesion_HC_inst_sub-237_0.nii.gz",
                   "/home/mwynen/synthetic_images/label_lesion_HC_inst_sub-268_0.nii.gz"]
    for label_path in label_paths:
        label_data = nib.load(label_path).get_fdata()
        label_tensor = torch.from_numpy(label_data).unsqueeze(0)  # Add channel dimension
        data = {"label": label_tensor}
        for i in range(2):
            transformed_data = transform(data)
            transformed_data = train_transforms(transformed_data)
            synthetic_image = transformed_data["image"]
            print(synthetic_image.shape)
            # Save or visualize synthetic_image as needed
            synthetic_image = synthetic_image.numpy()  # Convert to numpy for saving or visualization
            synthetic_image = np.squeeze(synthetic_image)  # Remove channel dimension if needed

            label = transformed_data["label"]
            label = label.numpy()
            label = np.squeeze(label)

            # nib.save(nib.Nifti1Image(synthetic_image, np.eye(4)), label_path.replace('labels.nii.gz', f'synthetic_image_{i}.nii.gz'))
            # nib.save(nib.Nifti1Image(label, np.eye(4)), label_path.replace('labels.nii.gz', f'synthetic_label_{i}.nii.gz'))
            # nib.save(nib.Nifti1Image(synthetic_image, np.eye(4)), label_path.replace('labels_instances.nii.gz', f'synthetic_image_{i}.nii.gz'))
            # nib.save(nib.Nifti1Image(label, np.eye(4)), label_path.replace('labels_instances.nii.gz', f'synthetic_label_{i}.nii.gz'))
            nib.save(nib.Nifti1Image(synthetic_image, np.eye(4)), label_path.replace('.nii.gz', f'_synthetic_image_{i}.nii.gz'))
            nib.save(nib.Nifti1Image(label, np.eye(4)), label_path.replace('.nii.gz', f'_synthetic_label_{i}.nii.gz'))
