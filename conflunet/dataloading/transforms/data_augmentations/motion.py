import numpy as np
import torch
import SimpleITK as sitk

from monai.transforms import RandomizableTransform
from monai.config import KeysCollection
from typing import Sequence, Union, Tuple, Optional, Dict

class RandMotiond(RandomizableTransform):
    def __init__(
        self,
        keys: KeysCollection,
        degrees: Union[float, Tuple[float, float]] = 10,
        translation: Union[float, Tuple[float, float]] = 10,  # in mm
        num_transforms: int = 2,
        prob: float = 0.1,
        perturbation: float = 0.3,
        image_interpolation: str = 'linear',
    ):
        super().__init__(prob=prob)
        self.keys = keys
        if isinstance(degrees, (float, int)):
            self.degrees_range = (-degrees, degrees)
        else:
            self.degrees_range = degrees
        if isinstance(translation, (float, int)):
            self.translation_range = (-translation, translation)
        else:
            self.translation_range = translation
        self.num_transforms = num_transforms
        self.perturbation = perturbation
        self.image_interpolation = image_interpolation

    def randomize(self) -> None:
        self._times = self._get_times(self.num_transforms, self.perturbation)
        self._degrees = np.random.uniform(
            self.degrees_range[0], self.degrees_range[1], size=(self.num_transforms, 3)
        )
        self._translation = np.random.uniform(
            self.translation_range[0], self.translation_range[1], size=(self.num_transforms, 3)
        )

    def _get_times(self, num_transforms, perturbation):
        step = 1 / (num_transforms + 1)
        times = np.arange(0, 1, step)[1:]
        noise = np.random.uniform(-step * perturbation, step * perturbation, size=num_transforms)
        return np.clip(times + noise, 0, 1)

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if isinstance(img, torch.Tensor):
                img_np = img.numpy()
            else:
                img_np = img

            n_channels = img_np.shape[0] if img_np.ndim == 4 else 1
            result_img = None
            out_image = []
            for i in range(n_channels):
                img_sitk = sitk.GetImageFromArray(img_np[i].transpose(2, 1, 0))
                # img_sitk.CopyInformation(sitk.Image(img_np[i].shape[::-1], sitk.sitkFloat32))

                transforms = self._get_rigid_transforms(img_sitk)
                images = self._resample_images(img_sitk, transforms)

                spectra = []
                for im in images:
                    arr = sitk.GetArrayFromImage(im).transpose()
                    tensor = torch.from_numpy(arr)
                    spectrum = torch.fft.fftn(tensor, dim=(-3, -2, -1), norm='ortho')
                    spectra.append(spectrum)

                self._sort_spectra(spectra, self._times)

                result_spectrum = torch.zeros_like(spectra[0])
                last_index = result_spectrum.shape[2]
                indices = (last_index * self._times).astype(int).tolist()
                indices.append(last_index)

                ini = 0
                for spectrum, fin in zip(spectra, indices):
                    result_spectrum[..., ini:fin] = spectrum[..., ini:fin]
                    ini = fin

                result_img = torch.fft.ifftn(result_spectrum, dim=(-3, -2, -1), norm='ortho').real.float()

            assert result_img is not None, "No valid image was generated."
            d[key] = result_img.unsqueeze(0) if n_channels == 1 else torch.stack(out_image, dim=0)

        return d

    def _get_rigid_transforms(self, image: sitk.Image) -> Sequence[sitk.Euler3DTransform]:
        center = np.array(image.GetSize()) / 2.0
        center_phys = image.TransformContinuousIndexToPhysicalPoint(center.tolist())
        transforms = []
        for deg, trans in zip(self._degrees, self._translation):
            radians = np.radians(deg).tolist()
            t = sitk.Euler3DTransform()
            t.SetCenter(center_phys)
            t.SetRotation(*radians)
            t.SetTranslation(trans.tolist())
            transforms.append(t)
        return transforms

    def _resample_images(
        self, image: sitk.Image, transforms: Sequence[sitk.Euler3DTransform]
    ) -> Sequence[sitk.Image]:
        images = [image]
        for transform in transforms:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)
            resampler.SetInterpolator(sitk.sitkLinear if self.image_interpolation == 'linear' else sitk.sitkNearestNeighbor)
            resampler.SetTransform(transform)
            images.append(resampler.Execute(image))
        return images

    def _sort_spectra(self, spectra: Sequence[torch.Tensor], times: np.ndarray):
        num_spectra = len(spectra)
        if np.any(times > 0.5):
            idx = np.where(times > 0.5)[0][0]
        else:
            idx = num_spectra - 1
        spectra[0], spectra[idx] = spectra[idx], spectra[0]



if __name__ == '__main__':
    from monai.transforms import Compose, RandGaussianSmoothd, OneOf, RandAffined, Rand3DElasticd, RandFlipd, RandBiasFieldd, RandGaussianNoised, \
        RandAdjustContrastd, EnsureChannelFirstd, LoadImaged

    image_path = '/home/mwynen/data/cusl_wml/all/images/sub-001_ses-01_reg-T2starw_FLAIR.nii.gz'
    out_path = '/home/mwynen/sub-001_with_motion.nii.gz'
    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        RandMotiond(
            keys=["image"],
            degrees=5,
            translation=5,
            num_transforms=1,
            prob=1.0,
            perturbation=0.3,
            image_interpolation='linear'
        ),
    ])

    data = {"image": image_path}
    import nibabel as nib
    transformed_data = transform(data)
    transformed_image = transformed_data["image"]
    transformed_image = transformed_image.squeeze().numpy()
    nib.save(nib.Nifti1Image(transformed_image, np.eye(4)), out_path)
