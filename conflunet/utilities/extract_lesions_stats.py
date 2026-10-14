import numpy as np
import SimpleITK as sitk
import pandas as pd
from typing import Union, Tuple, List, Dict, Any

from utils import cast_label_image_to_int


def extract_label_stats(
    label_image: Union[str, sitk.Image], background_value: int = 0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract statistics for each blob in the given label image and summary statistics.

    :param label_image: The input label image, either as a SimpleITK Image object or a file path.
    :type label_image: Union[str, sitk.Image]
    :param background_value: The value to be considered as background in the label image, defaults to 0.
    :type background_value: int, optional
    :return: A tuple containing a list of dictionaries with statistics for each label and a dictionary with summary statistics.
    :rtype: Tuple[List[Dict[str, Any]], Dict[str, Any]]
    """
    # read the label image
    label_image = (
        label_image
        if isinstance(label_image, sitk.Image)
        else sitk.ReadImage(label_image)
    )
    # Check if the image has a single foreground value (a mask) and instantiate different lesions if necessary
    if (
        len(np.unique(sitk.GetArrayFromImage(label_image))) == 2
    ):  # Assuming the image has only two unique values: background and foreground
        connected_components = sitk.ConnectedComponentImageFilter()
        connected_components.SetBackgroundValue(background_value)
        label_image = connected_components.Execute(label_image)

    # Check if the image is float type or int type bigger than need it for the number od labels
    label_image = cast_label_image_to_int(label_image)
    # extract the label stats
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.SetComputePerimeter(True)
    stats.SetComputeFeretDiameter(True)
    stats.SetComputeOrientedBoundingBox(True)
    stats.Execute(label_image)
    # Populate dictionary with each lesions and it features from LabelShapeStatisticsImageFilter
    lesions_stats = [
        {
            "Lesion": label,
            "BoundingBox": stats.GetBoundingBox(label),
            "Centroid": stats.GetCentroid(label),
            "Elongation": stats.GetElongation(label),
            "EquivalentEllipsoidDiameter": stats.GetEquivalentEllipsoidDiameter(label),
            "EquivalentSphericalPerimeter": stats.GetEquivalentSphericalPerimeter(
                label
            ),
            "EquivalentSphericalRadius": stats.GetEquivalentSphericalRadius(label),
            "Flatness": stats.GetFlatness(label),
            "PrincipalMoments": stats.GetPrincipalMoments(label),
            "NumberOfPixels": stats.GetNumberOfPixels(label),
            "NumberOfPixelsOnBorder": stats.GetNumberOfPixelsOnBorder(label),
            "Perimeter": stats.GetPerimeter(label),
            "PerimeterOnBorder": stats.GetPerimeterOnBorder(label),
            "PerimeterOnBorderRatio": stats.GetPerimeterOnBorderRatio(label),
            "PhysicalSize": stats.GetPhysicalSize(label),
            "PrincipalAxes": stats.GetPrincipalAxes(label),
            "Roundness": stats.GetRoundness(label),
            "FeretDiameter": stats.GetFeretDiameter(label),
            "OrientedBoundingBoxVertices": stats.GetOrientedBoundingBoxVertices(label),
            "OrientedBoundingBoxOrigin": stats.GetOrientedBoundingBoxOrigin(label),
            "OrientedBoundingBoxSize": stats.GetOrientedBoundingBoxSize(label),
            "OrientedBoundingBoxDirection": stats.GetOrientedBoundingBoxDirection(
                label
            ),
        }
        for label in stats.GetLabels()
    ]
    physical_sizes = [lesion["PhysicalSize"] for lesion in lesions_stats]
    if physical_sizes:
        summary_stats = {
            "N_lesions": len(lesions_stats),
            "Total_volume": np.sum(physical_sizes),
            "Lesion_volume_mean": np.mean(physical_sizes),
            "Lesion_volume_std": np.std(physical_sizes),
            "Lesion_volume_min": np.min(physical_sizes),
            "Lesion_volume_max": np.max(physical_sizes),
            "Lesion_volume_median": np.median(physical_sizes),
            "Lesion_iqr": np.percentile(physical_sizes, 75)
            - np.percentile(physical_sizes, 25),
        }
    else:  # if no lesions are found
        summary_stats = {
            "N_lesions": 0,
            "Total_volume": 0,
            "Lesion_volume_mean": 0,
            "Lesion_volume_std": 0,
            "Lesion_volume_min": 0,
            "Lesion_volume_max": 0,
            "Lesion_volume_median": 0,
            "Lesion_iqr": 0,
        }

    return lesions_stats, summary_stats


if __name__ == "__main__":
    import os
    import glob
    import pandas as pd

    main_dir = "/media/secondary/ConfluentLesions_INsIDER"
    main_labels_dir = os.path.join(main_dir, "*.nii.gz")
    save_instances_path = os.path.join(main_dir, "all_instances_stats.csv")
    save_summary_path = os.path.join(main_dir, "summary_instances_stats.csv")
    lesions_mask_paths = glob.glob(main_labels_dir)
    lesions_list = []
    lesions_summary_list = []

    for mask in lesions_mask_paths:
        # Extract filename and folder name
        filename = os.path.basename(mask)
        subject_id = filename.split("_")[0]
        session = filename.split("_")[1]
        folder_name = main_dir
        # Switch for assigning site_exp based on folder_name
        site_exp = "Basel_INsIDER"

        print(f"Processing file: {filename} in {folder_name}")
        label_stats, summary_stats = extract_label_stats(mask)
        print(f"Number of lesions: {summary_stats['N_lesions']}")
        print(f"Total lesion volume: {summary_stats['Total_volume']:.2f}")
        print("---")

        # Create a dictionary with common metadata
        metadata = {
            "subject_id": subject_id,
            "TP": session,
            "filename": filename,
            "site_exp": site_exp,
        }

        # Update label_stats and summary_stats with metadata
        # Update each dictionary in label_stats with metadata
        for stats in label_stats:
            stats.update(metadata)
        lesions_list += label_stats
        # Update summary_stats with metadata
        summary_stats.update(metadata)
        lesions_summary_list.append(summary_stats)

    lesions_df = pd.DataFrame(lesions_list)
    lesions_summary_df = pd.DataFrame(lesions_summary_list)

    lesions_df.to_csv(save_instances_path, index=False)
    lesions_summary_df.to_csv(save_summary_path, index=False)
