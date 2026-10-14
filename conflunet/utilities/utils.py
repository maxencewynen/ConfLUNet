import os
import subprocess
import shutil
import tempfile
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from typing import Tuple
import pandas as pd


def cast_label_image_to_int(label_image: sitk.Image) -> sitk.Image:
    """
    Cast a SimpleITK label image to an appropriate integer type.

    This function determines the most suitable integer type for the label image
    based on its minimum and maximum values, and casts the image to that type.

    :param label_image: The input label image to be cast.
    :type label_image: sitk.Image
    :return: The label image cast to an appropriate integer type.
    :rtype: sitk.Image
    """
    if label_image.GetPixelID() in [sitk.sitkFloat32, sitk.sitkFloat64]:
        print(
            "Warning: Label image is float type. Casting to int for connected component analysis."
        )

    min_value = sitk.GetArrayViewFromImage(label_image).min()
    max_value = sitk.GetArrayViewFromImage(label_image).max()

    if min_value >= 0:
        if max_value <= 255:
            cast_type = sitk.sitkUInt8
        elif max_value <= 65535:
            cast_type = sitk.sitkUInt16
        else:
            cast_type = sitk.sitkUInt32
    else:
        if min_value >= -128 and max_value <= 127:
            cast_type = sitk.sitkInt8
        elif min_value >= -32768 and max_value <= 32767:
            cast_type = sitk.sitkInt16
        else:
            cast_type = sitk.sitkInt32

    if label_image.GetPixelID() != cast_type:
        return sitk.Cast(label_image, cast_type)
    else:
        return label_image


def generate_dataset_json(
    output_folder: str,
    channel_names: dict,
    labels: dict,
    num_training_cases: int,
    file_ending: str,
    regions_class_order: Tuple[int, ...] = None,
    dataset_name: str = None,
    reference: str = None,
    release: str = None,
    license: str = None,
    description: str = None,
    overwrite_image_reader_writer: str = None,
    **kwargs,
):
    """
    Generates a dataset.json file in the output folder

    Parameters
    ----------
    output_folder : str
        folder where the dataset.json should be saved
    channel_names : dict
        Channel names must map the index to the name of the channel, example:
        {
            0: 'T1',
            1: 'CT'
        }
        Note that the channel names may influence the normalization scheme!! Learn more in the documentation.
    labels : dict
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training:
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }

        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!
    num_training_cases : int
        is used to double check all cases are there!
    file_ending : str
        needed for finding the files correctly. IMPORTANT! File endings must match between images and
        segmentations!
    regions_class_order : Tuple[int, ...]
        If you have defined regions (see above), you need to specify the order in which they should be
        processed. This is important because it determines the color in the 2d/3d visualizations.
    dataset_name : str, optional
        dataset name, by default None
    reference : str, optional
        reference, by default None
    release : str, optional
        release, by default None
    license : str, optional
        license, by default None
    description : str, optional
        description, by default None
    overwrite_image_reader_writer : str, optional
        If you need a special IO class for your dataset you can derive it from
        BaseReaderWriter, place it into nnunet.imageio and reference it here by name
    **kwargs
        whatever you put here will be placed in the dataset.json as well

    """
    has_regions: bool = any(
        [isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()]
    )
    if has_regions:
        assert regions_class_order is not None, (
            f"You have defined regions but regions_class_order is not set. "
            f"You need that."
        )
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for l in labels.keys():
        value = labels[l]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[l] = value
        else:
            labels[l] = int(labels[l])

    dataset_json = {
        "channel_names": channel_names,  # previously this was called 'modality'. I didn't like this so this is
        # channel_names now. Live with it.
        "labels": labels,
        "numTraining": num_training_cases,
        "file_ending": file_ending,
    }

    if dataset_name is not None:
        dataset_json["name"] = dataset_name
    if reference is not None:
        dataset_json["reference"] = reference
    if release is not None:
        dataset_json["release"] = release
    if license is not None:
        dataset_json["licence"] = license
    if description is not None:
        dataset_json["description"] = description
    if overwrite_image_reader_writer is not None:
        dataset_json["overwrite_image_reader_writer"] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json["regions_class_order"] = regions_class_order

    dataset_json.update(kwargs)

    save_json(dataset_json, join(output_folder, "dataset.json"), sort_keys=False)


class SCRIPTS:
    dockerName = "petermcgor/ants:2.3.1"
    registrationSyN = "antsRegistrationSyN.sh"
    applyTransform = "antsApplyTransforms"
    dockerSynthStrip = "freesurfer/synthstrip"


imgs_folder_dck = "/data"
out_folder_dck = "/out"  # imgs_folder_dck if out_folder is None else '/out'


def synth_strip(
    imgs_folder: str,
    input_img: str,
    out_folder: str,
    out_img: str = None,
    b: int = 1,
    save_brain_mask=False,
):
    """
    Use FreeSurfer's SynthStrip to skull strip an image. Requires docker and a GPU.

    Parameters
    ----------
    imgs_folder : str
        Folder where the input image is located.
    input_img : str
        Input image to skull strip.
    out_folder : str
        Folder where the output image will be saved.
    out_img : str, optional
        Output image name. If None, the same as the input image is used.
    b : int, optional
        Brain extraction threshold. Default is 1.

    Returns
    -------
    None
    """
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    out_img = input_img if out_img is None else out_img
    brain_mask = out_img.split(".")[0] + "_brain-mask.nii.gz"

    # Sometimes dockers mounts (config) can (should) struggle with symbolic links or already mounted filesystems
    # Working on the temp should help but there is a overload moving data
    tmp_dir = tempfile.TemporaryDirectory()
    shutil.copy(os.path.join(imgs_folder, input_img), tmp_dir.name)

    subprocess.run(
        [
            "docker",
            "run",
            "-v",
            tmp_dir.name + ":" + imgs_folder_dck,
            "-v",
            tmp_dir.name + ":" + out_folder_dck,
            "--gpus",
            "device=0",
            SCRIPTS.dockerSynthStrip,
            "-i",
            os.path.join(imgs_folder_dck, input_img),
            "-o",
            os.path.join(out_folder_dck, out_img),
            "-m",
            os.path.join(out_folder_dck, brain_mask),
            "-b",
            str(b),
        ]
    )
    shutil.copy(os.path.join(tmp_dir.name, out_img), out_folder)
    if save_brain_mask:
        shutil.copy(os.path.join(tmp_dir.name, brain_mask), out_folder)
    os.remove(os.path.join(tmp_dir.name, input_img))

    # remove out
