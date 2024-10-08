"""
This code is completely copied from a later version of the MONAI library. The original code can be found at
https://github.com/Project-MONAI/MONAI/blob/59a7211070538586369afd4a01eca0a7fe2e742e/monai/transforms/intensity/array.py
https://github.com/Project-MONAI/MONAI/blob/59a7211070538586369afd4a01eca0a7fe2e742e/monai/transforms/intensity/dictionary.py
"""

from monai.transforms.transform import Transform, RandomizableTransform
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_to_tensor
from typing import Sequence, Mapping, Hashable, Any
import numpy as np
from monai.transforms import MapTransform
from monai.config import KeysCollection, NdarrayOrTensor
from monai.data import get_track_meta


class AdjustContrast(Transform):
    """
    Changes image intensity with gamma transform. Each pixel/voxel intensity is updated as::

        x = ((x - min) / intensity_range) ^ gamma * intensity_range + min

    Args:
        gamma: gamma value to adjust the contrast as function.
        invert_image: whether to invert the image before applying gamma augmentation. If True, multiply all intensity
            values with -1 before the gamma transform and again after the gamma transform. This behaviour is mimicked
            from `nnU-Net <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        retain_stats: if True, applies a scaling factor and an offset to all intensity values after gamma transform to
            ensure that the output intensity distribution has the same mean and standard deviation as the intensity
            distribution of the input. This behaviour is mimicked from `nnU-Net
            <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, gamma: float, invert_image: bool = False, retain_stats: bool = False) -> None:
        if not isinstance(gamma, (int, float)):
            raise ValueError(f"gamma must be a float or int number, got {type(gamma)} {gamma}.")
        self.gamma = gamma
        self.invert_image = invert_image
        self.retain_stats = retain_stats

    def __call__(self, img: NdarrayOrTensor, gamma=None) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        gamma: gamma value to adjust the contrast as function.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        gamma = gamma if gamma is not None else self.gamma

        if self.invert_image:
            img = -img

        if self.retain_stats:
            mn = img.mean()
            sd = img.std()

        epsilon = 1e-7
        img_min = img.min()
        img_range = img.max() - img_min
        ret: NdarrayOrTensor = ((img - img_min) / float(img_range + epsilon)) ** gamma * img_range + img_min

        if self.retain_stats:
            # zero mean and normalize
            ret = ret - ret.mean()
            ret = ret / (ret.std() + 1e-8)
            # restore old mean and standard deviation
            ret = sd * ret + mn

        if self.invert_image:
            ret = -ret

        return ret


class RandAdjustContrast(RandomizableTransform):
    """
    Randomly changes image intensity with gamma transform. Each pixel/voxel intensity is updated as:

        x = ((x - min) / intensity_range) ^ gamma * intensity_range + min

    Args:
        prob: Probability of adjustment.
        gamma: Range of gamma values.
            If single number, value is picked from (0.5, gamma), default is (0.5, 4.5).
        invert_image: whether to invert the image before applying gamma augmentation. If True, multiply all intensity
            values with -1 before the gamma transform and again after the gamma transform. This behaviour is mimicked
            from `nnU-Net <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        retain_stats: if True, applies a scaling factor and an offset to all intensity values after gamma transform to
            ensure that the output intensity distribution has the same mean and standard deviation as the intensity
            distribution of the input. This behaviour is mimicked from `nnU-Net
            <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
    """

    backend = AdjustContrast.backend

    def __init__(
        self,
        prob: float = 0.1,
        gamma: Sequence[float] | float = (0.5, 4.5),
        invert_image: bool = False,
        retain_stats: bool = False,
    ) -> None:
        RandomizableTransform.__init__(self, prob)

        if isinstance(gamma, (int, float)):
            if gamma <= 0.5:
                raise ValueError(
                    f"if gamma is a number, must greater than 0.5 and value is picked from (0.5, gamma), got {gamma}"
                )
            self.gamma = (0.5, gamma)
        elif len(gamma) != 2:
            raise ValueError("gamma should be a number or pair of numbers.")
        else:
            self.gamma = (min(gamma), max(gamma))

        self.gamma_value: float = 1.0
        self.invert_image: bool = invert_image
        self.retain_stats: bool = retain_stats

        self.adjust_contrast = AdjustContrast(
            self.gamma_value, invert_image=self.invert_image, retain_stats=self.retain_stats
        )

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.gamma_value = self.R.uniform(low=self.gamma[0], high=self.gamma[1])

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()

        if not self._do_transform:
            return img

        if self.gamma_value is None:
            raise RuntimeError("gamma_value is not set, please call `randomize` function first.")

        return self.adjust_contrast(img, self.gamma_value)

class RandAdjustContrastd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandAdjustContrast`.
    Randomly changes image intensity with gamma transform. Each pixel/voxel intensity is updated as:

        `x = ((x - min) / intensity_range) ^ gamma * intensity_range + min`

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        prob: Probability of adjustment.
        gamma: Range of gamma values.
            If single number, value is picked from (0.5, gamma), default is (0.5, 4.5).
        invert_image: whether to invert the image before applying gamma augmentation. If True, multiply all intensity
            values with -1 before the gamma transform and again after the gamma transform. This behaviour is mimicked
            from `nnU-Net <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        retain_stats: if True, applies a scaling factor and an offset to all intensity values after gamma transform to
            ensure that the output intensity distribution has the same mean and standard deviation as the intensity
            distribution of the input. This behaviour is mimicked from `nnU-Net
            <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandAdjustContrast.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        gamma: tuple[float, float] | float = (0.5, 4.5),
        invert_image: bool = False,
        retain_stats: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.adjuster = RandAdjustContrast(gamma=gamma, prob=1.0, invert_image=invert_image, retain_stats=retain_stats)
        self.invert_image = invert_image

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ):
        super().set_random_state(seed, state)
        self.adjuster.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random gamma value
        self.adjuster.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.adjuster(d[key], randomize=False)
        return d