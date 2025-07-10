import numpy as np
import torch
from typing import Callable
from copy import deepcopy
from monai.data.meta_tensor import MetaTensor
import torch.nn.functional as F
import os
import nibabel as nib
from monai.transforms import MapTransform
from monai.config import KeysCollection
from copy import copy

from conflunet.dataloading.transforms.data_augmentations.labelstoimage import LESION_LABELS


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


class CleanLabelsd(MapTransform):
    """
    A MONAI transform to clean label maps by removing specified lesion labels.
    """
    def __init__(self, keys: KeysCollection,
                 label_key='label',
                 instance_seg_key='instance_seg',
                 seg_key='seg',
                 brainmask_key='brainmask',
                 lesion_labels=LESION_LABELS,
                 allow_missing_keys=True):
        super().__init__(keys)
        self.keys = keys
        self.label_key = label_key
        self.instance_seg_key = instance_seg_key
        self.seg_key = seg_key
        self.brainmask_key = brainmask_key
        self.lesion_labels = lesion_labels
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data):
        d = dict(data)
        assert self.label_key in d, f"Key '{self.label_key}' not found in data."
        array = d[self.label_key]

        if isinstance(array, torch.Tensor):
            array = array.cpu().numpy()

        cleaned_array = np.where(np.isin(array, self.lesion_labels, invert=True), 0, array)
        d[self.instance_seg_key] = MetaTensor(cleaned_array)
        d[self.seg_key] = MetaTensor((cleaned_array > 0).astype(np.float32))
        d[self.brainmask_key] = MetaTensor((array > 0).astype(np.float32))

        # del d[self.label_key]

        return d