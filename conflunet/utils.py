import numpy as np
import torch
from typing import Callable, Tuple
from copy import deepcopy
from monai.data.meta_tensor import MetaTensor
import torch.nn.functional as F
import os
import nibabel as nib
from monai.transforms import MapTransform
from monai.config import KeysCollection
from postprocess import remove_small_lesions_from_instance_segmentation
from scipy.ndimage import center_of_mass


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


class BinarizeInstancesd(MapTransform):
    """
    Binarize the values of specified keys in the input dictionary.
    Args:
        keys (list): List of keys corresponding to the data to be binarized.
        out_key (str, optional): Output key for the binarized data. Defaults to "label".
    Returns:
        dict: The input data dictionary with binarized values.

    Example:
        >>> binarizer = BinarizeInstancesd(keys=["image", "mask"], out_key="binarized")
        >>> data = {"image": np.random.rand(64, 64), "mask": np.random.rand(64, 64)}
        >>> transformed_data = binarizer(data)
    """

    def __init__(self, keys: list, out_key: str = "label"):
        """
        Initializes the BinarizeInstancesd transform with the specified keys and output key.

        Args:
            keys (list): List of keys corresponding to the data to be binarized.
            out_key (str, optional): Output key for the binarized data. Defaults to "label".
        """
        super().__init__(keys)
        self.keys = keys
        self.out_key = out_key

    def __call__(self, data: dict) -> dict:
        """
        Binarizes the values of specified keys in the input dictionary.
        Args:
            data (dict): Input data dictionary containing the keys to be binarized.
        Returns:
            dict: The input data dictionary with binarized values.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            out_key = self.out_key + "_" + key if len(self.keys) > 1 else self.out_key
            assert np.all(np.unique(data[key]) >= 0), "The input data should be non-negative."
            image = deepcopy(data[key])
            image[image > 0] = 1
            d[out_key] = image.astype(np.uint8)
        return d


def make_offset_matrices(data, sigma=2, voxel_size=(1, 1, 1), remove_small_lesions=False, l_min=14):
    # Define 3D Gaussian function
    def gaussian_3d(x, y, z, cx, cy, cz, sigma):
        return np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * sigma ** 2))

    data = np.squeeze(data)
    if remove_small_lesions:
        data = remove_small_lesions_from_instance_segmentation(data, voxel_size=voxel_size, l_min=l_min)

    heatmap = np.zeros_like(data, dtype=np.float32)
    offset_x = np.zeros_like(data, dtype=np.float32)
    offset_y = np.zeros_like(data, dtype=np.float32)
    offset_z = np.zeros_like(data, dtype=np.float32)

    # Create coordinate grids
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(data.shape[0]),
                                         np.arange(data.shape[1]),
                                         np.arange(data.shape[2]),
                                         indexing='ij')

    # Get all unique lesion IDs (excluding zero which is typically background)
    lesion_ids = np.unique(data[data!=0])

    # For each lesion id
    for lesion_id in lesion_ids:
        # Get binary mask for the current lesion
        mask = (data == lesion_id)

        # Compute the center of mass of the lesion
        cx, cy, cz = center_of_mass(mask)

        # Compute heatmap values using broadcasting
        current_gaussian = gaussian_3d(x_grid, y_grid, z_grid, cx, cy, cz, sigma)

        # Update heatmap with the maximum value encountered so far at each voxel
        heatmap = np.maximum(heatmap, current_gaussian)

        # Update offset matrices
        offset_x[mask] = cx - x_grid[mask]
        offset_y[mask] = cy - y_grid[mask]
        offset_z[mask] = cz - z_grid[mask]

    return np.expand_dims(heatmap, axis=0).astype(np.float32), \
        np.stack([offset_x, offset_y, offset_z], axis=0).astype(np.float32)


class LesionOffsetTransformd(MapTransform):
    """
    A MONAI transform to compute the offsets for each voxel from the center of mass of its lesion.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys=False, remove_small_lesions=False, l_min=14, sigma=2):
        """
        Args:
            key (str): the key corresponding to the desired data in the dictionary to apply the transformation.
        """
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if type(keys) == list and len(keys) > 1:
            raise Exception("This transform should only be used with 1 key.")
        self.remove_small_lesions = remove_small_lesions
        self.l_min = l_min
        self.sigma = sigma

    def __call__(self, data):
        d = dict(data)
        voxel_size = tuple(data[[k for k in list(data.keys()) if "_meta_dict" in k][0]]['pixdim'][1:4])
        for key in self.key_iterator(d):
            com_gt, com_reg = self.make_offset_matrices(d[key], voxel_size=voxel_size, sigma=self.sigma)
            d["center_heatmap"] = com_gt
            d["offsets"] = com_reg
            d["label"] = (d[key] > 0).astype(np.uint8)
        return d

    def make_offset_matrices(self, data, sigma=2, voxel_size=(1, 1, 1)):
        return make_offset_matrices(data,
                                    sigma=sigma,
                                    voxel_size=voxel_size,
                                    remove_small_lesions=self.remove_small_lesions,
                                    l_min=self.l_min)




