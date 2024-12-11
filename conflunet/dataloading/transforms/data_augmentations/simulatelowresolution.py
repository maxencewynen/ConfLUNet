"""
This code is completely copied from a later version of the MONAI library. The original code can be found at
https://github.com/Project-MONAI/MONAI/blob/59a7211070538586369afd4a01eca0a7fe2e742e/monai/transforms/intensity/array.py
https://github.com/Project-MONAI/MONAI/blob/59a7211070538586369afd4a01eca0a7fe2e742e/monai/transforms/intensity/dictionary.py
"""

import torch
import numpy as np
from typing import Sequence, Mapping, Hashable, Any, Union

from monai.utils import InterpolateMode
from monai.transforms import MapTransform
from monai.data.meta_tensor import MetaTensor
from monai.config import KeysCollection, NdarrayOrTensor
from monai.utils.type_conversion import convert_to_tensor
from monai.transforms.transform import RandomizableTransform
from monai.data.meta_obj import get_track_meta, set_track_meta
from monai.transforms.spatial.array import RandAffine, Affine, Resize


class RandSimulateLowResolution(RandomizableTransform):
    """
    Random simulation of low resolution corresponding to nnU-Net's SimulateLowResolutionTransform
    (https://github.com/MIC-DKFZ/batchgenerators/blob/7651ece69faf55263dd582a9f5cbd149ed9c3ad0/batchgenerators/transforms/resample_transforms.py#L23)
    First, the array/tensor is resampled at lower resolution as determined by the zoom_factor which is uniformly sampled
    from the `zoom_range`. Then, the array/tensor is resampled at the original resolution.
    """

    backend = Affine.backend

    def __init__(
        self,
        prob: float = 0.1,
        downsample_mode: Union[InterpolateMode, str] = InterpolateMode.NEAREST,
        upsample_mode: Union[InterpolateMode, str] = InterpolateMode.TRILINEAR,
        zoom_range: Sequence[float] = (0.5, 1.0),
        align_corners=False,
        device: Union[torch.device, None] = None,
    ) -> None:
        """
        Args:
            prob: probability of performing this augmentation
            downsample_mode: interpolation mode for downsampling operation
            upsample_mode: interpolation mode for upsampling operation
            zoom_range: range from which the random zoom factor for the downsampling and upsampling operation is
            sampled. It determines the shape of the downsampled tensor.
            align_corners: This only has an effect when downsample_mode or upsample_mode  is 'linear', 'bilinear',
                'bicubic' or 'trilinear'. Default: False
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            device: device on which the tensor will be allocated.

        """
        RandomizableTransform.__init__(self, prob)

        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode
        self.zoom_range = zoom_range
        self.align_corners = align_corners
        self.device = device
        self.zoom_factor = 1.0

    def randomize(self, data: Union[Any, None] = None) -> None:
        super().randomize(None)
        self.zoom_factor = self.R.uniform(self.zoom_range[0], self.zoom_range[1])
        if not self._do_transform:
            return None

    def __call__(self, img: torch.Tensor, randomize: bool = True) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            randomize: whether to execute `randomize()` function first, defaults to True.
        """
        if randomize:
            self.randomize()

        if self._do_transform:
            input_shape = img.shape[1:]
            target_shape = np.round(np.array(input_shape) * self.zoom_factor).astype(np.int_)

            resize_tfm_downsample = Resize(
                spatial_size=target_shape, size_mode="all", mode=self.downsample_mode, anti_aliasing=False
            )

            resize_tfm_upsample = Resize(
                spatial_size=input_shape,
                size_mode="all",
                mode=self.upsample_mode,
                anti_aliasing=False,
                align_corners=self.align_corners,
            )
            # temporarily disable metadata tracking, since we do not want to invert the two Resize functions during
            # post-processing
            original_tack_meta_value = get_track_meta()
            set_track_meta(False)

            img_downsampled = resize_tfm_downsample(img)
            img_upsampled = resize_tfm_upsample(img_downsampled)

            # reset metadata tracking to original value
            set_track_meta(original_tack_meta_value)

            # copy metadata from original image to down-and-upsampled image
            img_upsampled = MetaTensor(img_upsampled)
            img_upsampled.copy_meta_from(img)

            return img_upsampled

        else:
            return img


class RandSimulateLowResolutiond(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandSimulateLowResolution`.
    Random simulation of low resolution corresponding to nnU-Net's SimulateLowResolutionTransform
    (https://github.com/MIC-DKFZ/batchgenerators/blob/7651ece69faf55263dd582a9f5cbd149ed9c3ad0/batchgenerators/transforms/resample_transforms.py#L23)
    First, the array/tensor is resampled at lower resolution as determined by the zoom_factor which is uniformly sampled
    from the `zoom_range`. Then, the array/tensor is resampled at the original resolution.
    """

    backend = RandAffine.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        downsample_mode: Union[InterpolateMode, str] = InterpolateMode.NEAREST,
        upsample_mode: Union[InterpolateMode, str] = InterpolateMode.TRILINEAR,
        zoom_range=(0.5, 1.0),
        align_corners=False,
        allow_missing_keys: bool = False,
        device: Union[torch.device, None] = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            prob: probability of performing this augmentation
            downsample_mode: interpolation mode for downsampling operation
            upsample_mode: interpolation mode for upsampling operation
            zoom_range: range from which the random zoom factor for the downsampling and upsampling operation is
            sampled. It determines the shape of the downsampled tensor.
            align_corners: This only has an effect when downsample_mode or upsample_mode  is 'linear', 'bilinear',
                'bicubic' or 'trilinear'. Default: False
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            allow_missing_keys: don't raise exception if key is missing.
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`monai.transforms.compose.MapTransform`

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode
        self.zoom_range = zoom_range
        self.align_corners = align_corners
        self.device = device

        self.sim_lowres_tfm = RandSimulateLowResolution(
            prob=1.0,  # probability is handled by dictionary class
            downsample_mode=self.downsample_mode,
            upsample_mode=self.upsample_mode,
            zoom_range=self.zoom_range,
            align_corners=self.align_corners,
            device=self.device,
        )

    def set_random_state(
        self, seed: Union[int, None] = None, state: Union[np.random.RandomState, None] = None
    ):
        super().set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be transformed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
        """
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out: dict[Hashable, NdarrayOrTensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)

        for key in self.key_iterator(d):
            # do the transform
            if self._do_transform:
                d[key] = self.sim_lowres_tfm(d[key])  # type: ignore
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta(), dtype=torch.float32)
        return d