"""
This code is completely copied from a later version of the MONAI library. The original code can be found at
https://github.com/Project-MONAI/MONAI/blob/59a7211070538586369afd4a01eca0a7fe2e742e/monai/transforms/intensity/array.py
https://github.com/Project-MONAI/MONAI/blob/59a7211070538586369afd4a01eca0a7fe2e742e/monai/transforms/intensity/dictionary.py
"""

import numpy as np
import torch
from typing import Sequence, Mapping, Hashable, Any, Union

from monai.data import get_track_meta
from monai.transforms import MapTransform
from monai.utils.enums import TransformBackends
from monai.transforms.utils_pytorch_numpy_unification import clip
from monai.config import KeysCollection, DtypeLike, NdarrayOrTensor
from monai.transforms.transform import Transform, RandomizableTransform
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor


class ScaleIntensityFixedMean(Transform):
    """
    Scale the intensity of input image by ``v = v * (1 + factor)``, then shift the output so that the output image has the
    same mean as the input.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        factor: float = 0,
        preserve_range: bool = False,
        fixed_mean: bool = True,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        """
        Args:
            factor: factor scale by ``v = v * (1 + factor)``.
            preserve_range: clips the output array/tensor to the range of the input array/tensor
            fixed_mean: subtract the mean intensity before scaling with `factor`, then add the same value after scaling
                to ensure that the output has the same mean as the input.
            channel_wise: if True, scale on each channel separately. `preserve_range` and `fixed_mean` are also applied
                on each channel separately if `channel_wise` is True. Please ensure that the first dimension represents the
                channel of the image if True.
            dtype: output data type, if None, same as input image. defaults to float32.
        """
        self.factor = factor
        self.preserve_range = preserve_range
        self.fixed_mean = fixed_mean
        self.channel_wise = channel_wise
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor, factor=None) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        Args:
            img: the input tensor/array
            factor: factor scale by ``v = v * (1 + factor)``

        """

        factor = factor if factor is not None else self.factor

        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t = convert_to_tensor(img, track_meta=False)
        ret: NdarrayOrTensor
        if self.channel_wise:
            out = []
            for d in img_t:
                if self.preserve_range:
                    clip_min = d.min()
                    clip_max = d.max()

                if self.fixed_mean:
                    mn = d.mean()
                    d = d - mn

                out_channel = d * (1 + factor)

                if self.fixed_mean:
                    out_channel = out_channel + mn

                if self.preserve_range:
                    out_channel = clip(out_channel, clip_min, clip_max)

                out.append(out_channel)
            ret = torch.stack(out)
        else:
            if self.preserve_range:
                clip_min = img_t.min()
                clip_max = img_t.max()

            if self.fixed_mean:
                mn = img_t.mean()
                img_t = img_t - mn

            ret = img_t * (1 + factor)

            if self.fixed_mean:
                ret = ret + mn

            if self.preserve_range:
                ret = clip(ret, clip_min, clip_max)

        ret = convert_to_dst_type(ret, dst=img, dtype=self.dtype or img_t.dtype)[0]
        return ret


class RandScaleIntensityFixedMean(RandomizableTransform):
    """
    Randomly scale the intensity of input image by ``v = v * (1 + factor)`` where the `factor`
    is randomly picked. Subtract the mean intensity before scaling with `factor`, then add the same value after scaling
    to ensure that the output has the same mean as the input.
    """

    backend = ScaleIntensityFixedMean.backend

    def __init__(
        self,
        prob: float = 0.1,
        factors: Union[Sequence[float], float] = 0,
        fixed_mean: bool = True,
        preserve_range: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        """
        Args:
            factors: factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            preserve_range: clips the output array/tensor to the range of the input array/tensor
            fixed_mean: subtract the mean intensity before scaling with `factor`, then add the same value after scaling
                to ensure that the output has the same mean as the input.
            channel_wise: if True, scale on each channel separately. `preserve_range` and `fixed_mean` are also applied
            on each channel separately if `channel_wise` is True. Please ensure that the first dimension represents the
            channel of the image if True.
            dtype: output data type, if None, same as input image. defaults to float32.

        """
        RandomizableTransform.__init__(self, prob)
        if isinstance(factors, (int, float)):
            self.factors = (min(-factors, factors), max(-factors, factors))
        elif len(factors) != 2:
            raise ValueError("factors should be a number or pair of numbers.")
        else:
            self.factors = (min(factors), max(factors))
        self.factor = self.factors[0]
        self.fixed_mean = fixed_mean
        self.preserve_range = preserve_range
        self.dtype = dtype

        self.scaler = ScaleIntensityFixedMean(
            factor=self.factor, fixed_mean=self.fixed_mean, preserve_range=self.preserve_range, dtype=self.dtype
        )

    def randomize(self, data: Union[Any, None] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.factor = self.R.uniform(low=self.factors[0], high=self.factors[1])

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()

        if not self._do_transform:
            return convert_data_type(img, dtype=self.dtype)[0]

        return self.scaler(img, self.factor)


class RandScaleIntensityFixedMeand(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandScaleIntensity`.
    Subtract the mean intensity before scaling with `factor`, then add the same value after scaling
    to ensure that the output has the same mean as the input.
    """

    backend = RandScaleIntensityFixedMean.backend

    def __init__(
        self,
        keys: KeysCollection,
        factors: Union[Sequence[float], float],
        fixed_mean: bool = True,
        preserve_range: bool = False,
        prob: float = 0.1,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            factors: factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            preserve_range: clips the output array/tensor to the range of the input array/tensor
            fixed_mean: subtract the mean intensity before scaling with `factor`, then add the same value after scaling
                to ensure that the output has the same mean as the input.
            channel_wise: if True, scale on each channel separately. `preserve_range` and `fixed_mean` are also applied
            on each channel separately if `channel_wise` is True. Please ensure that the first dimension represents the
            channel of the image if True.
            dtype: output data type, if None, same as input image. defaults to float32.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.fixed_mean = fixed_mean
        self.preserve_range = preserve_range
        self.scaler = RandScaleIntensityFixedMean(
            factors=factors, fixed_mean=self.fixed_mean, preserve_range=preserve_range, dtype=dtype, prob=1.0
        )

    def set_random_state(
        self, seed: Union[int, None] = None, state: Union[np.random.RandomState, None] = None
    ):
        super().set_random_state(seed, state)
        self.scaler.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random scale factor
        self.scaler.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key], randomize=False)
        return d