#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import multiprocessing
import shutil
from time import sleep
from typing import Tuple, Union
from wave import Error

from conflunet.preprocessing.utils import crop_to_nonzero, create_center_heatmap_from_instance_seg, \
    get_confluent_instances_classes, get_small_object_classes, merge_maps

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from tqdm import tqdm

import nnunetv2
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets, get_identifiers_from_splitted_dataset_folder, \
    create_lists_from_splitted_dataset_folder


class InstanceSegProcessor(DefaultPreprocessor):
    def __init__(self, verbose: bool = True,
                 inference : bool = False,
                 add_center_heatmap_in_npz: bool = True,
                 center_heatmap_sigma: int = 2,
                 add_small_object_classes_in_npz: bool = False,
                 small_objects_thresholds: int = 100,
                 add_confluent_instances_in_npz: bool = False,
                 output_dir_for_inference: str = None,
                 synthetic: bool = False
                 ):
        """
        :param verbose:
        :param add_center_heatmap_in_npz: Whether to add a matrix with center heatmaps in the npz file
        :param add_small_object_classes_in_npz: Whether to add a matrix with small object classes in the npz file.
            (label = 1 if the object is not small, 2 if it is small and 0 if it is not a lesion)
        :param small_objects_thresholds: Small objects are defined as objects with a volume (in voxels)
            smaller than this threshold
        :param add_confluent_instances_in_npz: Whether to add a matrix with confluent lesions in the npz file
            (label = 1 if the lesion is not confluent, 2 if it is confluent and 0 if it is not a lesion)
        """
        super(InstanceSegProcessor, self).__init__(verbose)
        self.inference = inference
        self.add_center_heatmap_in_npz = add_center_heatmap_in_npz if not self.inference else False
        self.center_heatmap_sigma = center_heatmap_sigma
        self.add_small_object_classes_in_npz = add_small_object_classes_in_npz if not self.inference else False
        self.small_objects_thresholds = small_objects_thresholds
        self.add_confluent_instances_in_npz = add_confluent_instances_in_npz if not self.inference else False
        self.output_dir_for_inference = output_dir_for_inference
        self.has_warned_about_1000_plus_instance_ids = False
        self.synthetic = synthetic
        if self.inference:
            assert self.output_dir_for_inference is not None, "If inference is True, output_dir_for_inference must be " \
                                                              "specified. Got None."
            assert isdir(self.output_dir_for_inference), "output_dir_for_inference must be a directory. Got %s" % \
                                                            self.output_dir_for_inference

    def modify_instance_seg_fn(self, instance_seg: np.ndarray) -> np.ndarray:
        """
        Maps instance ids to a continuous range starting from 1
        :param instance_seg: instance segmentation
        :return: instance segmentation with continuous instance ids starting from 1
        """
        unique_values = np.unique(instance_seg)
        assert all([i >= 0 for i in unique_values]), "Instance segmentation must not contain negative values"
        assert unique_values[0] == 0, "Instance segmentation must have a background value of 0"
        if np.max(unique_values) > 1000 and not self.has_warned_about_1000_plus_instance_ids:
            print("[WARNING] Instance segmentation contains values greater than 1000. These will be considered as "
                  "unsplittable lesions (i.e. they will be taken into account only for semantic segmentation, "
                  "not for instance segmentation).")
            print("[WARNING] If you want to change this behavior, please modify the modify_instance_seg_fn method ")
            print("[WARNING] This warning will only be displayed once.")
            self.has_warned_about_1000_plus_instance_ids = True
        mapping = {i: j + 1 for j, i in enumerate(unique_values[1:])}
        mapping.update({i: 1000 + j for j, i in enumerate([i for i in unique_values[1:] if i >= 1000])})
        mapping[0] = 0
        return np.vectorize(mapping.get)(instance_seg)

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        instance_seg = seg
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. \
            Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            instance_seg = np.copy(seg)
            seg = (np.copy(seg) > 0).astype(np.int8)  # seg is binary, instance seg is not
            if np.max(instance_seg) > 127:
                instance_seg = instance_seg.astype(np.int16)
            else:
                instance_seg = instance_seg.astype(np.int8)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if has_seg:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
            instance_seg = instance_seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, instance_seg, bbox = crop_to_nonzero(data, seg,
                                                        instance_seg)  # crop_to_nonzero has been modified, look in utils.py
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        old_shape = data.shape[1:]
        if not self.synthetic:
            data = self._normalize(data, seg, configuration_manager,
                                   plans_manager.foreground_intensity_properties_per_channel)

            # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
            #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
            data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        else:
            data = configuration_manager.resampling_fn_seg(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing) if has_seg else None
        instance_seg = configuration_manager.resampling_fn_seg(instance_seg, new_shape, original_spacing,
                                                               target_spacing) if has_seg else None
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                              verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
            instance_seg = self.modify_instance_seg_fn(instance_seg)

            if np.max(seg) > 127:
                seg = seg.astype(np.int16)
            else:
                seg = seg.astype(np.int8)
            if np.max(instance_seg) > 127:
                instance_seg = instance_seg.astype(np.int16)
            else:
                instance_seg = instance_seg.astype(np.int8)

        if self.add_center_heatmap_in_npz:
            # add center heatmaps
            center_heatmap = create_center_heatmap_from_instance_seg(instance_seg, sigma=self.center_heatmap_sigma)
        else:
            center_heatmap = None

        if self.add_small_object_classes_in_npz:
            # add small object classes
            small_object_classes = get_small_object_classes(instance_seg, threshold=self.small_objects_thresholds)
        else:
            small_object_classes = None
        if self.add_confluent_instances_in_npz:
            # add confluent instances
            confluent_instances = get_confluent_instances_classes(instance_seg)
        else:
            confluent_instances = None

        return data, seg, instance_seg, (center_heatmap, small_object_classes, confluent_instances)

    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager, dataset_json: Union[dict, str]):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # load image(s)
        data, data_properties = rw.read_images(image_files)

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        data, seg, instance_seg, other = self.run_case_npy(data, seg, data_properties, plans_manager,
                                                           configuration_manager, dataset_json)
        return data, seg, instance_seg, other, data_properties

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str]):
        data, seg, instance_seg, other, properties = self.run_case(image_files, seg_file, plans_manager,
                                                                   configuration_manager, dataset_json)
        # print('dtypes', data.dtype, seg.dtype)
        center_heatmap, small_object_classes, confluent_instances = other
        kwargs = {'data': data, 'seg': seg, 'instance_seg': instance_seg}
        if center_heatmap is not None:
            kwargs['center_heatmap'] = center_heatmap
        if small_object_classes is not None and confluent_instances is not None:
            kwargs['small_objects_and_confluent_instances_classes'] = merge_maps(small_object_classes,
                                                                                 confluent_instances)
        if small_object_classes is not None:
            kwargs['small_object_classes'] = small_object_classes
        if confluent_instances is not None:
            kwargs['confluent_instances'] = confluent_instances

        np.savez_compressed(output_filename_truncated + '.npz', **kwargs)
        write_pickle(properties, output_filename_truncated + '.pkl')

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
        """
        data identifier = configuration name in plans. EZ.
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
        if self.verbose:
            print(configuration_manager)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)

        if isdir(output_directory):
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)

        dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)
        # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
        # output_filenames_truncated = [join(output_directory, i) for i in identifiers]

        # multiprocessing magic.
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            remaining = list(range(len(dataset)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]

            for k in dataset.keys():
                r.append(p.starmap_async(self.run_case_save,
                                         ((join(output_directory, k), dataset[k]['images'], dataset[k]['label'],
                                           plans_manager, configuration_manager,
                                           dataset_json),)))

            with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                           'OK jokes aside.\n'
                                           'One of your background processes is missing. This could be because of '
                                           'an error (look for an error message) or because it was killed '
                                           'by your OS due to running out of RAM. If you don\'t see '
                                           'an error message, out of RAM is likely the problem. In that case '
                                           'reducing the number of workers might help')
                    done = [i for i in remaining if r[i].ready()]
                    # get done so that errors can be raised
                    _ = [r[i].get() for i in done]
                    for _ in done:
                        r[_].get()  # allows triggering errors
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

    def run_for_inference(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
                          num_processes: int, input_dir: str):
        """
        This is the same as run but for inference. The difference is that we do not need stuff needed for training,
        and we do not need to save the gt segmentations. We also add an input_dir argument to specify where the
        raw input data is located. Temporary preprocessed files will be stored in the output_dir.
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
        if self.verbose:
            print(configuration_manager)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        output_directory = self.output_dir_for_inference
        maybe_mkdir_p(output_directory)

        if isdir(output_directory):
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)

        dataset = get_filenames_of_test_images(input_dir, dataset_json)
        # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
        # output_filenames_truncated = [join(output_directory, i) for i in identifiers]

        # multiprocessing magic.
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            remaining = list(range(len(dataset)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]

            for k in dataset.keys():
                r.append(p.starmap_async(self.run_case_save,
                                         ((join(output_directory, k), dataset[k]['images'], None,
                                           plans_manager, configuration_manager,
                                           dataset_json),)))

            with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                           'OK jokes aside.\n'
                                           'One of your background processes is missing. This could be because of '
                                           'an error (look for an error message) or because it was killed '
                                           'by your OS due to running out of RAM. If you don\'t see '
                                           'an error message, out of RAM is likely the problem. In that case '
                                           'reducing the number of workers might help')
                    done = [i for i in remaining if r[i].ready()]
                    # get done so that errors can be raised
                    _ = [r[i].get() for i in done]
                    for _ in done:
                        r[_].get()  # allows triggering errors
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)


def get_filenames_of_test_images(input_dir: str, dataset_json: dict):
    if dataset_json is None:
        raise ValueError("dataset_json must be provided")

    if 'dataset' in dataset_json.keys():
        dataset = dataset_json['dataset']
        for k in dataset.keys():
            dataset[k]['images'] = [os.path.abspath(join(input_dir, i)) if not os.path.isabs(i) else i for
                                    i in dataset[k]['images']]
    else:
        identifiers = get_identifiers_from_splitted_dataset_folder(input_dir,
                                                                   dataset_json['file_ending'])
        images = create_lists_from_splitted_dataset_folder(input_dir,
                                                           dataset_json['file_ending'], identifiers)
        dataset = {i: {'images': im} for i, im in zip(identifiers, images)}
    return dataset
