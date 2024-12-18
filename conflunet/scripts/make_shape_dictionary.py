import os
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
from os.path import join as pjoin
from scipy.ndimage import binary_closing

from nnunetv2.paths import nnUNet_preprocessed

from conflunet.utilities.planning_and_configuration import load_dataset_and_configuration


def extract_patch(instance_seg, instance_id):
    """
    Extract the shape of an instance from the instance segmentation.
    Returns a binary patch of the instance.
    """
    binary_mask = instance_seg == instance_id

    mask_indices = np.where(binary_mask)

    min_x, max_x = mask_indices[0].min(), mask_indices[0].max()
    min_y, max_y = mask_indices[1].min(), mask_indices[1].max()
    min_z, max_z = mask_indices[2].min(), mask_indices[2].max()

    patch = binary_mask[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]

    return patch  # maybe add some info about the borders or the center of the patch?


def extract_image_patch(image, instance_seg, instance_id):
    """
    Extract the shape of an instance from the instance segmentation.
    Returns a binary patch of the instance.
    """
    binary_mask = instance_seg == instance_id

    mask_indices = np.where(binary_mask)

    min_x, max_x = mask_indices[0].min(), mask_indices[0].max()
    min_y, max_y = mask_indices[1].min(), mask_indices[1].max()
    min_z, max_z = mask_indices[2].min(), mask_indices[2].max()

    image_patch = image[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]

    return image_patch


def make_shape_dictionary(dataset_id: int) -> dict:
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(dataset_id)
    preprocessed_dir = pjoin(nnUNet_preprocessed, dataset_name, "nnUNetPlans_3d_fullres")
    output_dir = pjoin(nnUNet_preprocessed, dataset_name, "shapes")
    os.makedirs(output_dir, exist_ok=True)
    voxel_size = configuration.spacing

    shapes = []
    for filename in os.listdir(preprocessed_dir):
        if not filename.endswith(".npz"):
            continue
        case_id = filename.split(".npz")[0]
        print(f"Processing {case_id}...")
        data = np.load(pjoin(preprocessed_dir, filename))

        image = data["data"][0]
        instance_seg = data["instance_seg"][0]
        small_objects = data["small_object_classes"][0]
        confluent_instances = data["confluent_instances"][0]
        for instance_id in tqdm(np.unique(instance_seg)):
            if instance_id == 0:
                continue

            binary_patch = extract_patch(instance_seg, instance_id)
            image_patch = extract_image_patch(image, instance_seg, instance_id)
            # out_file = pjoin(output_dir, f"{case_id}_{instance_id}.npy")
            # np.save(out_file, binary_patch)
            # out_file = pjoin(output_dir, f"{case_id}_{instance_id}.nii.gz")
            # nib.save(nib.Nifti1Image(binary_patch.astype(np.int16), np.eye(4)), out_file)

            closed_patch = binary_closing(binary_patch)
            out_file = pjoin(output_dir, f"{case_id}_{instance_id}.npy")
            np.save(out_file, closed_patch)
            image_out_file = pjoin(output_dir, f"{case_id}_{instance_id}_image.npy")
            np.save(image_out_file, image_patch)
            # out_file = pjoin(output_dir, f"{case_id}_{instance_id}_closed.nii.gz")
            # nib.save(nib.Nifti1Image(closed_patch.astype(np.int16), np.eye(4)), out_file)

            # find any voxel that is part of the instance
            coord_x, coord_y, coord_z = np.where(instance_seg == instance_id)
            coord = (coord_x[0], coord_y[0], coord_z[0])
            is_too_small = small_objects[coord] == 2
            is_confluent = confluent_instances[coord] == 2
            shapes.append({
                'case_id': case_id,
                'instance_id': int(instance_id),
                "file": out_file,
                "image_file": image_out_file,
                "is_too_small": int(is_too_small),
                "is_confluent": int(is_confluent),
                "volume": int(binary_patch.sum()) * np.prod(voxel_size),
            })

    with open(pjoin(output_dir, "shapes.json"), "w") as f:
        json.dump(shapes, f, indent=4)

    return shapes


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id", type=int)
    args = parser.parse_args()

    shapes = make_shape_dictionary(args.dataset_id)
    pprint(shapes, indent=4)
