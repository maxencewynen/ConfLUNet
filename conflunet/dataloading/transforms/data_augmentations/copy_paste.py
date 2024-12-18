import json
import torch
import numpy as np
from os.path import join as pjoin
from os.path import exists as pexists
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter, binary_dilation

from monai.data import get_track_meta
from monai.transforms import MapTransform
from monai.utils.enums import TransformBackends
from monai.config import KeysCollection, NdarrayOrTensor
from monai.utils.type_conversion import convert_to_tensor
from monai.transforms.transform import Transform, RandomizableTransform


def sample_lesion_patch(path_to_shapes: str):
    """
    Sample a lesion patch.
    """
    if not pexists(path_to_shapes):
        raise FileNotFoundError(f"Shapes directory not found at {path_to_shapes}")

    # load json file with shapes
    with open(pjoin(path_to_shapes, 'shapes.json'), 'r') as f:
        shapes = json.load(f)

    # sample a random shape
    shape_infos = np.random.choice(shapes)
    binary_patch = np.load(shape_infos['file'])
    return binary_patch


def sample_lesion_patch_with_image(path_to_shapes: str):
    """
    Sample a lesion patch.
    """
    if not pexists(path_to_shapes):
        raise FileNotFoundError(f"Shapes directory not found at {path_to_shapes}")

    # load json file with shapes
    with open(pjoin(path_to_shapes, 'shapes.json'), 'r') as f:
        shapes = json.load(f)

    # sample a random shape
    shape_infos = np.random.choice(shapes)
    binary_patch = np.load(shape_infos['file'])
    image_patch = np.load(shape_infos['image_file'])
    return binary_patch, image_patch


def get_random_location_inside_mask(mask_to_sample, binary_patch, iterations=20):
    """
    Get a random location for the patch inside the valid mask while ensuring boundaries are respected.
    """
    # TODO: check if random location is valid (inside WM)
    # TODO: Find random location to create confluence
    if iterations == 0:
        return None

    # Get mask indices where the patch can be pasted
    indices = np.where(mask_to_sample == 1)
    random_index = np.random.choice(len(indices[0]))
    random_coords = [indices[i][random_index] for i in range(3)]

    # Patch dimensions
    patch_shape = binary_patch.shape
    half_shape = [s // 2 for s in patch_shape]

    # Calculate patch boundaries
    start_coords = [random_coords[i] - half_shape[i] for i in range(3)]
    end_coords = [start_coords[i] + patch_shape[i] for i in range(3)]

    # Check if patch stays inside the mask and image
    for i in range(3):
        if start_coords[i] < 0 or end_coords[i] > mask_to_sample.shape[i]:
            print("Patch out of bounds. Sampling new location.")
            return get_random_location_inside_mask(mask_to_sample, binary_patch, iterations - 1)

    # Validate that the patch doesn't overlap invalid regions
    patch_region = mask_to_sample[start_coords[0]:end_coords[0],
                   start_coords[1]:end_coords[1],
                   start_coords[2]:end_coords[2]]
    if patch_region.shape != binary_patch.shape or np.any(patch_region == 0):
        print("Patch intersects with undesired regions. Sampling new location.")
        return get_random_location_inside_mask(mask_to_sample, binary_patch, iterations - 1)

    return random_coords

# def find_closest_lesion(instance_seg, binary_patch, coords):
#     """
#     Find the closest lesion to the new binary patch using a distance transform.
#     """
#     # Generate binary mask of existing lesions
#     lesion_mask = instance_seg > 0
#
#     # Compute distance transform to find distances from the patch boundary to existing lesions
#     distance = distance_transform_edt(~lesion_mask)
#
#     # Define the patch region
#     patch_shape = binary_patch.shape
#     half_shape = [s // 2 for s in patch_shape]
#     start_coords = [coords[i] - half_shape[i] for i in range(3)]
#     end_coords = [start_coords[i] + patch_shape[i] for i in range(3)]
#
#     # Ensure bounds
#     for i in range(3):
#         if start_coords[i] < 0 or end_coords[i] > instance_seg.shape[i]:
#             raise ValueError("Patch region out of bounds!")
#
#     # Extract the patch region from the distance map
#     region = tuple(slice(start_coords[i], end_coords[i]) for i in range(3))
#     patch_distance = distance[region]
#
#     # Find coordinates of the minimum distance
#     min_distance_coords = np.unravel_index(np.argmin(patch_distance), patch_distance.shape)
#     closest_coords = [start_coords[i] + min_distance_coords[i] for i in range(3)]
#
#     # Identify the closest lesion ID at these coordinates
#     closest_lesion_id = instance_seg[tuple(closest_coords)]
#     if closest_lesion_id == 0:
#         return None  # No valid lesion found
#     return closest_lesion_id
#
#
# def paste_from_coords(image, instance_seg, binary_patch, coords):
#     """
#     Paste the binary patch to the image at the given coordinates,
#     using the avg and std intensities of the closest lesion.
#     """
#     # Find the closest lesion and its avg and std intensity
#     closest_lesion_id = find_closest_lesion(instance_seg, binary_patch, coords)
#     if closest_lesion_id is not None:
#         lesion_mask = instance_seg == closest_lesion_id
#         avg_intensity = image[lesion_mask].mean()
#         std_intensity = image[lesion_mask].std()
#     else:
#         # Default to global avg and std if no lesion is found
#         avg_intensity = image[instance_seg > 0].mean()
#         std_intensity = image[instance_seg > 0].std()
#
#     # Get the shape and half dimensions of the patch
#     patch_shape = binary_patch.shape
#     half_shape = [s // 2 for s in patch_shape]
#
#     # Calculate patch region coordinates
#     start_coords = [coords[i] - half_shape[i] for i in range(3)]
#     end_coords = [start_coords[i] + patch_shape[i] for i in range(3)]
#
#     # Ensure coordinates are within bounds
#     for i in range(3):
#         if start_coords[i] < 0 or end_coords[i] > image.shape[i]:
#             print("Invalid paste region. Skipping this patch.")
#             return image, instance_seg
#
#     # Build random intensities for the patch based on closest lesion
#     patch_intensity = binary_patch * np.random.normal(avg_intensity, std_intensity, size=binary_patch.shape)
#
#     # Paste only where binary_patch == 1
#     region = tuple(slice(start_coords[i], end_coords[i]) for i in range(3))
#     image_region = image[region]
#     mask = binary_patch == 1
#     image_region[mask] = patch_intensity[mask]
#
#     # Update the image and instance segmentation map
#     image[region] = image_region
#     instance_seg[region][mask] = (instance_seg.max() + 1)
#
#     return image, instance_seg


def paste_from_coords_copy_from_image_no_blurring(image, instance_seg, binary_patch, coords, image_patch):
    """
    Paste the binary patch to the image at the given coordinates, copying the intensity from the image_patch
    and changing intensity only where binary_patch == 1.
    """
    # Get the shape and half dimensions of the patch
    patch_shape = binary_patch.shape
    half_shape = [s // 2 for s in patch_shape]

    # Calculate patch region coordinates
    start_coords = [coords[i] - half_shape[i] for i in range(3)]
    end_coords = [start_coords[i] + patch_shape[i] for i in range(3)]

    # Ensure coordinates are within bounds
    for i in range(3):
        if start_coords[i] < 0 or end_coords[i] > image.shape[i]:
            print("Invalid paste region. Skipping this patch.")
            return image, instance_seg

    # Build random intensities for the patch
    patch_intensity = binary_patch * image_patch

    # Paste only where binary_patch == 1
    region = tuple(slice(start_coords[i], end_coords[i]) for i in range(3))
    image_region = image[region]
    mask = binary_patch == 1
    image_region[mask] = patch_intensity[mask]

    # Update the image
    image[region] = image_region

    # Update the instance segmentation map
    instance_seg[region][mask] = (instance_seg.max() + 1)

    return image, instance_seg


# def paste_from_coords(image, instance_seg, binary_patch, coords, image_patch):
#     """
#     Paste the binary patch to the image at the given coordinates, copying the intensity from the image_patch
#     and changing intensity only where binary_patch == 1.
#     """
#     binary_patch = np.pad(binary_patch, pad_width=1, mode='constant', constant_values=0)
#     image_patch = np.pad(image_patch, pad_width=1, mode='constant', constant_values=0)
#
#     # Build random intensities for the patch
#     patch_intensity = binary_patch * image_patch
#
#     # Create a mask for the border
#     dilated_patch = binary_dilation(binary_patch)
#     border_mask = np.logical_xor(dilated_patch, binary_patch)
#
#     # Apply Gaussian blur to the outer border
#     blurred_image_patch = gaussian_filter(patch_intensity, sigma=1)
#     patch_intensity[border_mask] = blurred_image_patch[border_mask]
#
#     # Get the shape and half dimensions of the patch
#     patch_shape = dilated_patch.shape
#     half_shape = [s // 2 for s in patch_shape]
#
#     # Calculate patch region coordinates
#     start_coords = [coords[i] - half_shape[i] for i in range(3)]
#     end_coords = [start_coords[i] + patch_shape[i] for i in range(3)]
#
#     # Ensure coordinates are within bounds
#     for i in range(3):
#         if start_coords[i] < 0 or end_coords[i] > image.shape[i]:
#             print("Invalid paste region. Skipping this patch.")
#             return image, instance_seg
#
#     # Paste only where binary_patch == 1
#     region = tuple(slice(start_coords[i], end_coords[i]) for i in range(3))
#     image_region = image[region]
#     mask = binary_patch == 1
#     image_region[dilated_patch] = patch_intensity[dilated_patch]
#
#     # Update the image
#     image[region] = image_region
#
#     # Update the instance segmentation map
#     instance_seg[region][mask] = (instance_seg.max() + 1)
#
#     return image, instance_seg


if __name__ == "__main__":
    data = np.load("/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans_3d_fullres/sub-038_ses-01.npz")
    modified_instance_seg = instance_seg = data["instance_seg"][0]
    modified_image = image = data["data"][0]
    seg = data["seg"][0]
    print(instance_seg.shape)
    lesion_voxels = image[instance_seg > 0]
    avg_lesion_intensity = lesion_voxels.mean()
    std_lesion_intensity = lesion_voxels.std()

    for i in range(120):
        # binary_patch = sample_lesion_patch(path_to_shapes = "/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/shapes")
        binary_patch, image_patch = sample_lesion_patch_with_image(path_to_shapes = "/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/shapes")
        mask_to_sample = (seg >= 0) & (modified_instance_seg == 0)

        random_coords = get_random_location_inside_mask(mask_to_sample=mask_to_sample, binary_patch=binary_patch)
        if random_coords is None:
            print("Could not find a valid location.")
            break
        print(random_coords)

        # modified_image, modified_instance_seg = paste_from_coords(image=modified_image, instance_seg=modified_instance_seg,
        #                                                           binary_patch=binary_patch, coords=random_coords)
        modified_image, modified_instance_seg = paste_from_coords_copy_from_image_no_blurring(image=modified_image, instance_seg=modified_instance_seg,
                                                                  binary_patch=binary_patch, coords=random_coords, image_patch=image_patch)
        # modified_image, modified_instance_seg = paste_from_coords(image=modified_image, instance_seg=modified_instance_seg,
        #                                                           binary_patch=binary_patch, coords=random_coords, image_patch=image_patch)

    import nibabel as nib
    nib.save(nib.Nifti1Image(modified_image, affine=np.eye(4)),
             "/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans_3d_fullres/sub-038_ses-01_image_plus_copy_paste.nii.gz")
    nib.save(nib.Nifti1Image(modified_instance_seg, affine=np.eye(4)),
            "/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans_3d_fullres/sub-038_ses-01_instance_seg_plus_copy_paste.nii.gz")
    nib.save(nib.Nifti1Image(seg, affine=np.eye(4)),
            "/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans_3d_fullres/sub-038_ses-01_seg.nii.gz")
    nib.save(nib.Nifti1Image(((seg >= 0) & (modified_instance_seg == 0)).astype(np.uint8), affine=np.eye(4)),
            "/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans_3d_fullres/sub-038_ses-01_mask_to_sample.nii.gz")



