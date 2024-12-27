import json
import warnings
from typing import Tuple, Union, Any, Optional, Mapping, Hashable, Dict, List, Sequence

import torch
import numpy as np
from os.path import join as pjoin
from os.path import exists as pexists
from scipy.ndimage import gaussian_filter, binary_dilation, binary_closing, binary_erosion

from monai.transforms import MapTransform
from monai.config import KeysCollection, NdarrayOrTensor
from monai.transforms.transform import RandomizableTransform


def sample_lesion_patch_with_image(shapes = Union[str, Dict[str, Any]]):
    """
    Sample a lesion patch.
    """
    if isinstance(shapes, str):
        if not pexists(shapes):
            raise FileNotFoundError(f"Shapes directory not found at {shapes}")

        # load json file with shapes
        with open(pjoin(shapes, 'shapes.json'), 'r') as f:
            shapes = json.load(f)

    # sample a random shape
    shape_infos = np.random.choice(shapes)
    binary_patch = np.load(shape_infos['file'])
    image_patch = np.load(shape_infos['image_file'])
    return binary_patch, image_patch


def precompute_lesion_borders(orig_lesion_seg):
    eroded_orig_lesion_seg = binary_erosion(orig_lesion_seg, iterations=1)
    lesion_borders = np.logical_xor(eroded_orig_lesion_seg, orig_lesion_seg)
    return eroded_orig_lesion_seg, lesion_borders


def get_random_location_inside_mask(mask_to_sample, binary_patch, location_probability, eroded_orig_lesion_seg, lesion_borders, max_iterations=30):
    """
    Get a random location for the patch inside the valid mask while ensuring boundaries are respected.
    """
    # Get mask indices where the patch can be pasted
    indices = np.where(mask_to_sample == 1)
    associated_probabilities = location_probability[indices]
    associated_probabilities /= associated_probabilities.sum()  # Normalize probabilities

    patch_shape = binary_patch.shape
    half_shape = [s // 2 for s in patch_shape]

    actual_mask_to_sample = binary_closing(mask_to_sample | lesion_borders)

    for _ in range(max_iterations):
        random_index = np.random.choice(len(indices[0]), p=associated_probabilities)
        random_coords = [indices[i][random_index] for i in range(3)]

        # Calculate patch boundaries
        start_coords = [random_coords[i] - half_shape[i] for i in range(3)]
        end_coords = [start_coords[i] + patch_shape[i] for i in range(3)]

        # Check if patch stays inside the mask and image
        if any(start < 0 or end > dim for start, end, dim in zip(start_coords, end_coords, mask_to_sample.shape)):
            continue

        # Validate that the patch doesn't overlap invalid regions
        patch_region = actual_mask_to_sample[start_coords[0]:end_coords[0],
                                             start_coords[1]:end_coords[1],
                                             start_coords[2]:end_coords[2]]
        eroded_orig_lesion_seg_region = eroded_orig_lesion_seg[start_coords[0]:end_coords[0],
                                                               start_coords[1]:end_coords[1],
                                                               start_coords[2]:end_coords[2]]

        if (patch_region.shape == binary_patch.shape and not np.any(patch_region == 0) and
                not np.any(patch_region & eroded_orig_lesion_seg_region)):
            return random_coords

    return None


def paste_from_coords_copy_from_image_no_blurring(image, instance_seg, binary_patch, coords, image_patch, **kwargs):
    """
    Paste the binary patch to the image at the given coordinates, copying the intensity from the image_patch
    and changing intensity only where binary_patch == 1.
    """
    print("Pasting from coords, no blurring")
    # Get the shape and half dimensions of the patch
    patch_shape = binary_patch.shape
    half_shape = [s // 2 for s in patch_shape]

    # Calculate patch region coordinates
    start_coords = [coords[i] - half_shape[i] for i in range(3)]
    end_coords = [start_coords[i] + patch_shape[i] for i in range(3)]

    # Ensure coordinates are within bounds
    for i in range(3):
        if start_coords[i] < 0 or end_coords[i] > image.shape[i]:
            return image, instance_seg

    # Build random intensities for the patch
    patch_intensity = binary_patch * image_patch

    # Paste only where binary_patch == 1
    region = tuple(slice(start_coords[i], end_coords[i]) for i in range(3))
    image_region = image[:, region[0], region[1], region[2]]
    mask = binary_patch == 1
    image_region[:, mask] = patch_intensity[:, mask]

    # Update the image
    image[:, region[0], region[1], region[2]] = image_region

    # Update the instance segmentation map
    instance_seg[region][mask] = (instance_seg.max() + 1)

    return image, instance_seg


def paste_from_coords_gaussian_blurring_outer_borders(
        image,
        instance_seg,
        binary_patch,
        coords,
        image_patch,
        valid_pasting_mask=1,
        avg_lesion_intensities: Tuple = (0,),
        std_lesion_intensities: Tuple = (0,)
):
    """
    Paste the binary patch to the image at the given coordinates, copying the intensity from the image_patch
    and changing intensity only where binary_patch == 1.
    """
    if image.shape[0] != image_patch.shape[0]:
        raise ValueError("The number of channels in the image does not match the number channels in the image patch.")
    else:
        this_lesion_avg_intensities = (image_patch * binary_patch).mean(axis=(1, 2, 3))
        for i in range(image.shape[0]):
            if this_lesion_avg_intensities[i] < (avg_lesion_intensities[i] - 1.5 * std_lesion_intensities[i]):
                low = avg_lesion_intensities[i] - 1.5 * std_lesion_intensities[i] - this_lesion_avg_intensities[i]
                high = avg_lesion_intensities[i] - this_lesion_avg_intensities[i]
                image_patch[i][binary_patch] += np.random.uniform(low, high)
            elif this_lesion_avg_intensities[i] > (avg_lesion_intensities[i] + 1.5 * std_lesion_intensities[i]):
                low = avg_lesion_intensities[i] - this_lesion_avg_intensities[i]
                high = (avg_lesion_intensities[i] + 1.5 * std_lesion_intensities[i]) - this_lesion_avg_intensities[i]
                image_patch[i][binary_patch] += np.random.uniform(low, high)

    pad_width = 2
    pad_widths = [(pad_width, pad_width)] * image.ndim
    pad_widths[0] = (0, 0)
    binary_patch = np.pad(binary_patch, pad_width=2, mode='constant', constant_values=0)
    image_patch = np.pad(image_patch, pad_width=pad_widths, mode='constant', constant_values=0)

    # Get the shape and half dimensions of the patch
    patch_shape = binary_patch.shape
    half_shape = [s // 2 for s in patch_shape]

    # Calculate patch region coordinates
    start_coords = [coords[i] - half_shape[i] for i in range(3)]
    end_coords = [start_coords[i] + patch_shape[i] for i in range(3)]

    # Ensure coordinates are within bounds
    for i in range(3):
        if start_coords[i] < 0 or end_coords[i] > image.shape[i]:
            return image, instance_seg

    # Build random intensities for the patch
    patch_intensity = binary_patch * image_patch

    # Paste only where binary_patch == 1
    region = tuple(slice(start_coords[i], end_coords[i]) for i in range(3))
    image_region = image[:, region[0], region[1], region[2]]
    mask = binary_patch == 1
    if type(patch_intensity[:, mask]) != type(image_region[:, mask]):
        patch_intensity = torch.from_numpy(patch_intensity).to(image_region.dtype)
    image_region[:, mask] = patch_intensity[:, mask]

    # Update the image
    image[:, region[0], region[1], region[2]] = image_region

    full_image_binary_mask = np.zeros_like(instance_seg)
    full_image_binary_mask[:, region[0], region[1], region[2]] = binary_patch
    full_image_binary_mask_dilated = binary_dilation(full_image_binary_mask, iterations=1)
    border_mask = np.logical_xor(full_image_binary_mask_dilated, full_image_binary_mask)

    # Update the instance segmentation map
    instance_seg[region][mask] = instance_seg.max() + 1

    # Apply Gaussian blur to the outer border
    image_blurred = gaussian_filter(image, sigma=5)
    # add a little bit of noise to the blurred image
    image_blurred = image_blurred + np.random.normal(0, 0.05, image_blurred.shape)
    if type(image[:, full_image_binary_mask_dilated]) != type(image_blurred[:, full_image_binary_mask_dilated]):
        image_blurred = torch.from_numpy(image_blurred).to(image.dtype)
    if type(valid_pasting_mask) != type(image_blurred[:, full_image_binary_mask_dilated]):
        valid_pasting_mask = torch.from_numpy(valid_pasting_mask).to(bool)
    image_blurred = torch.where(valid_pasting_mask, image_blurred, image)

    image[:, border_mask] = image_blurred[:, border_mask]

    return image, instance_seg


def paste_from_coords_gaussian_blurring_full(
        image,
        instance_seg,
        binary_patch,
        coords,
        image_patch,
        valid_pasting_mask=1,
        avg_lesion_intensities: Tuple = (0,),
        std_lesion_intensities: Tuple = (0,)
):
    """
    Paste the binary patch to the image at the given coordinates, copying the intensity from the image_patch
    and changing intensity only where binary_patch == 1.
    """
    if image.shape[0] != image_patch.shape[0]:
        raise ValueError("The number of channels in the image does not match the number channels in the image patch.")
    else:
        this_lesion_avg_intensities = (image_patch * binary_patch).mean(axis=(1, 2, 3))
        for i in range(image.shape[0]):
            if this_lesion_avg_intensities[i] < (avg_lesion_intensities[i] - 1.5 * std_lesion_intensities[i]):
                low = avg_lesion_intensities[i] - 1.5 * std_lesion_intensities[i] - this_lesion_avg_intensities[i]
                high = avg_lesion_intensities[i] - this_lesion_avg_intensities[i]
                image_patch[i][binary_patch] += np.random.uniform(low, high)
            elif this_lesion_avg_intensities[i] > (avg_lesion_intensities[i] + 1.5 * std_lesion_intensities[i]):
                low = avg_lesion_intensities[i] - this_lesion_avg_intensities[i]
                high = (avg_lesion_intensities[i] + 1.5 * std_lesion_intensities[i]) - this_lesion_avg_intensities[i]
                image_patch[i][binary_patch] += np.random.uniform(low, high)

    pad_width = 2
    pad_widths = [(pad_width, pad_width)] * image.ndim
    pad_widths[0] = (0, 0)
    binary_patch = np.pad(binary_patch, pad_width=2, mode='constant', constant_values=0)
    image_patch = np.pad(image_patch, pad_width=pad_widths, mode='constant', constant_values=0)

    # Get the shape and half dimensions of the patch
    patch_shape = binary_patch.shape
    half_shape = [s // 2 for s in patch_shape]

    # Calculate patch region coordinates
    start_coords = [coords[i] - half_shape[i] for i in range(3)]
    end_coords = [start_coords[i] + patch_shape[i] for i in range(3)]

    # Ensure coordinates are within bounds
    for i in range(3):
        if start_coords[i] < 0 or end_coords[i] > image.shape[i+1]:
            return image, instance_seg

    patch_intensity = binary_patch * image_patch

    # Paste only where binary_patch == 1
    region = tuple(slice(start_coords[i], end_coords[i]) for i in range(3))
    image_region = image[:, region[0], region[1], region[2]]
    mask = binary_patch == 1
    if type(patch_intensity[:, mask]) != type(image_region[:, mask]):
        patch_intensity = torch.from_numpy(patch_intensity).to(image_region.dtype)
    image_region[:, mask] = patch_intensity[:, mask]

    # Update the image
    image[:, region[0], region[1], region[2]] = image_region

    full_image_binary_mask = np.zeros_like(instance_seg)
    full_image_binary_mask[region] = binary_patch
    full_image_binary_mask_dilated = binary_dilation(full_image_binary_mask, iterations=2)

    # Update the instance segmentation map
    instance_seg[region][mask] = instance_seg.max() + 1

    # Apply Gaussian blur to the outer border
    image_blurred = gaussian_filter(image, sigma=1)
    # # add a little bit of noise to the blurred image
    # image_blurred = image_blurred + np.random.normal(0, 0.05, image_blurred.shape)
    if type(image[:, full_image_binary_mask_dilated]) != type(image_blurred[:, full_image_binary_mask_dilated]):
        image_blurred = torch.from_numpy(image_blurred).to(image.dtype)
    if type(valid_pasting_mask) != type(image_blurred[:, full_image_binary_mask_dilated]):
        valid_pasting_mask = torch.from_numpy(valid_pasting_mask).to(bool)
    image_blurred = torch.where(valid_pasting_mask, image_blurred, image)

    image[:, full_image_binary_mask_dilated] = image_blurred[:, full_image_binary_mask_dilated]

    return image, instance_seg


class RandCopyPasted(RandomizableTransform, MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 allow_missing_keys: bool = False,
                 image_key: str = "img",
                 instance_seg_key: str = "instance_seg",
                 seg_key: str = "seg",
                 paste_region_mask_keys: Union[str, Sequence[str]] = ("brainmask", "nawm"),
                 prob: float = 0.8,
                 n_objects_to_paste_range: Tuple[int, int] = (0, 1),
                 blend_mode: str = 'gaussian',
                 confluence_proportion: float = 0.1,
                 path_to_json: str = None,
                 ):
        """
            prob: probability of performing this augmentation
            n_objects_to_paste_range: The number of objects to paste in the image
            blend_mode: The blending mode to use for the pasted objects. Either ('none',
            'gaussian_outer_borders', 'gaussian').
            confluence_proportion: Proportion of pasted objects that should be confluent
            path_to_json: Path to the json file containing the shapes and information about the lesions
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.n_objects_to_paste_range = n_objects_to_paste_range
        self.n_objects_to_paste = None
        self.image_key = image_key
        self.instance_seg_key = instance_seg_key
        self.seg_key = seg_key
        if isinstance(paste_region_mask_keys, (list, tuple)):
            self.paste_region_mask_keys = paste_region_mask_keys
        else:
            self.paste_region_mask_keys = [paste_region_mask_keys]
        match blend_mode:
            case 'none':
                self.paste_fn = paste_from_coords_copy_from_image_no_blurring
            case 'gaussian_outer_borders':
                self.paste_fn = paste_from_coords_gaussian_blurring_outer_borders
            case 'gaussian':
                self.paste_fn = paste_from_coords_gaussian_blurring_full
            case _:
                raise ValueError(f"Invalid blend mode: {blend_mode}")
        self.confluence_proportion = confluence_proportion
        with open(path_to_json, 'r') as f:
            self.shapes = json.load(f)

    def randomize(self, randomize: Optional[bool] = None) -> None:
        super().randomize(None)
        self.n_objects_to_paste = self.R.uniform(self.n_objects_to_paste_range[0], self.n_objects_to_paste_range[1])
        if not self._do_transform:
            return None

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        """
        Args:
            randomize: whether to execute `randomize()` function first, defaults to True.
        """
        self.randomize()

        if not self._do_transform:
            return dict(data)

        instance_seg = data[self.instance_seg_key].squeeze()
        image = data[self.image_key]
        seg = data[self.seg_key].squeeze()
        paste_region_mask = np.array([1])
        for prmk in self.paste_region_mask_keys:
            if prmk in data and data[prmk] is not None:
                paste_region_mask = paste_region_mask & data[prmk].astype(bool)
        paste_region_mask = paste_region_mask.squeeze()

        modified_instance_seg = instance_seg
        n_objects = len(torch.unique(modified_instance_seg)) - 1
        modified_image = image
        orig_seg = modified_seg = seg
        lesion_intensity = modified_image[:, orig_seg.to(bool)]
        avg_lesion_intensities = lesion_intensity.mean(axis=1)
        std_lesion_intensities = lesion_intensity.std(axis=1)
        del lesion_intensity
        eroded_orig_lesion_seg, lesion_borders = precompute_lesion_borders(orig_seg)

        if not isinstance(paste_region_mask, int):
            paste_region_mask = paste_region_mask.astype(bool)

        added_lesions = 0
        n_objects_to_paste = int(self.n_objects_to_paste * n_objects)

        for i in range(n_objects_to_paste):
            mask_to_sample = (modified_seg >= 0) & (modified_instance_seg == 0) & paste_region_mask
            n_tries = 0
            random_coords = None

            close_to_lesion = binary_dilation(orig_seg, iterations=4)
            close_to_lesion = np.logical_xor(orig_seg > 0, close_to_lesion) & mask_to_sample
            far_from_lesion = ~close_to_lesion & mask_to_sample
            n_close_to_lesion = close_to_lesion.sum()
            n_far_from_lesion = far_from_lesion.sum()
            probability_close_to_lesion = self.confluence_proportion / n_close_to_lesion
            probability_far_from_lesion = (1 - self.confluence_proportion) / n_far_from_lesion
            assert np.isclose((n_close_to_lesion * probability_close_to_lesion) +
                              (n_far_from_lesion * probability_far_from_lesion), 1.0, atol=1e-7)

            location_probability = ((close_to_lesion * probability_close_to_lesion) +
                                    (far_from_lesion * probability_far_from_lesion))

            # We try to paste maximum 5 different lesions
            while n_tries < 5 and random_coords is None:
                binary_patch, image_patch = sample_lesion_patch_with_image(shapes=self.shapes)

                binary_patch = torch.from_numpy(binary_patch)
                image_patch = torch.from_numpy(image_patch).to(image.dtype)
                # We try to find a valid location to paste the lesion (max X locations)
                random_coords = get_random_location_inside_mask(mask_to_sample=mask_to_sample,
                                                                eroded_orig_lesion_seg = eroded_orig_lesion_seg,
                                                                lesion_borders = lesion_borders,
                                                                binary_patch=binary_patch,
                                                                location_probability=location_probability,
                                                                max_iterations=30)
                n_tries += 1

            if random_coords is None:
                continue

            added_lesions += 1
            modified_image, modified_instance_seg = self.paste_fn(
                image=modified_image,
                instance_seg=modified_instance_seg,
                binary_patch=binary_patch,
                coords=random_coords,
                image_patch=image_patch,
                valid_pasting_mask=paste_region_mask,
                avg_lesion_intensities=avg_lesion_intensities,
                std_lesion_intensities=std_lesion_intensities
            )

            modified_seg[modified_instance_seg > 0] = 1

        data[self.image_key] = modified_image.to(image.dtype).to(image.device)
        data[self.instance_seg_key] = modified_instance_seg.to(instance_seg.dtype).to(instance_seg.device).unsqueeze(0)
        data[self.seg_key] = modified_seg.to(seg.dtype).to(seg.device).unsqueeze(0)

        data = dict(data)

        return data


if __name__ == "__main__":
    pass
