from typing import Union
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json

from conflunet.preprocessing.preprocess import PlansManagerInstanceSeg, ConfigurationManagerInstanceSeg
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from nnunetv2.configuration import default_num_processes


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManagerInstanceSeg,
                                                                configuration_manager: ConfigurationManagerInstanceSeg,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes,
                                                                ):
    """
    Adapted from nnUNet's implementation to make it work on npz files
    https://github.com/MIC-DKFZ/nnUNet/blob/2eaa371317c55e22bf5166428315076a1ba9b7af/nnunetv2/inference/export_prediction.py
    """
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                                                         properties_dict[
                                                                             'shape_after_cropping_and_before_resampling'],
                                                                         current_spacing,
                                                                         [properties_dict['spacing'][i] for i in
                                                                          plans_manager.transpose_forward])
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    del predicted_logits
    segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(
                                                  label_manager.foreground_labels) < 255 else np.uint16)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    if return_probabilities:
        # revert cropping
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping


def convert_predicted_instance_segmentation_to_original_shape(instance_segmentation: Union[torch.Tensor, np.ndarray],
                                                              plans_manager: PlansManagerInstanceSeg,
                                                              configuration_manager: ConfigurationManagerInstanceSeg,
                                                              label_manager: LabelManager,
                                                              properties_dict: dict,
                                                              # return_probabilities: bool = False,
                                                              num_threads_torch: int = default_num_processes,
                                                              ):
    """
    Adapted from nnUNet's implementation to make it work on instance segmentation
    https://github.com/MIC-DKFZ/nnUNet/blob/2eaa371317c55e22bf5166428315076a1ba9b7af/nnunetv2/inference/export_prediction.py
    """
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]

    # here I replaced resampling_fn_probabilities with resampling_fn_seg to make it work on instance segmentation,
    # which have discrete values as opposed to probabilities which are continuous
    instance_segmentation = configuration_manager.resampling_fn_seg(instance_segmentation,
                                                                    properties_dict[
                                                                        'shape_after_cropping_and_before_resampling'],
                                                                    current_spacing,
                                                                    [properties_dict['spacing'][i] for i in
                                                                     plans_manager.transpose_forward])
    # Original code, commented out for reference
    # # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # # apply_inference_nonlin will convert to torch
    # predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    # del predicted_logits
    # segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(instance_segmentation, torch.Tensor):
        instance_segmentation = instance_segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(
                                                  label_manager.foreground_labels) < 255 else np.uint16)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = instance_segmentation
    del instance_segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    # Original code, commented out for reference
    # if return_probabilities:
    #     # revert cropping
    #     predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
    #                                                                              properties_dict[
    #                                                                                  'bbox_used_for_cropping'],
    #                                                                              properties_dict[
    #                                                                                  'shape_before_cropping'])
    #     predicted_probabilities = predicted_probabilities.cpu().numpy()
    #     # revert transpose
    #     predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
    #                                                                        plans_manager.transpose_backward])
    #     torch.set_num_threads(old_threads)
    #     return segmentation_reverted_cropping, predicted_probabilities
    # else:
    #     torch.set_num_threads(old_threads)
    #     return segmentation_reverted_cropping
    torch.set_num_threads(old_threads)
    return segmentation_reverted_cropping


if __name__ == '__main__':
    # test convert_predicted_instance_segmentation_to_original_shape
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    from nnunetv2.utilities.utils import load_pickle
    from conflunet.preprocessing.utils import save_nii
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    input_file = "/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans_3d_fullres/sub-038_ses-01.npz"
    output_file = input_file.replace(".npz", "_original_shape_and_spacing.nii.gz")

    pkl_file = "/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans_3d_fullres/sub-038_ses-01.pkl"
    properties_dict = load_pickle(pkl_file)

    data = np.load(input_file)
    instance_seg = data['instance_seg']

    plans_manager = PlansManagerInstanceSeg("/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans.json")
    configuration_manager = plans_manager.get_configuration("3d_fullres")

    dataset_json = "/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/dataset.json"
    dataset_json = load_json(dataset_json)
    label_manager = plans_manager.get_label_manager(dataset_json)
    postprocessed_instance_seg = \
        convert_predicted_instance_segmentation_to_original_shape(instance_seg, plans_manager, configuration_manager, label_manager, properties_dict)

    SimpleITKIO().write_seg(postprocessed_instance_seg, output_file, properties_dict)



