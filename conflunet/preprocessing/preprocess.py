# Preprocesses .nii or .nii.gz files (and their associated instance segmentations) to create 3D images in nnUNet format.
# The same preprocessing is applied as in the nnUNet preprocessing pipeline.
from typing import List, Type, Optional, Tuple, Union

from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json, isdir, maybe_mkdir_p
from nnunetv2.experiment_planning.plan_and_preprocess_api import plan_experiment_dataset
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.configuration import default_num_processes
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

from conflunet.preprocessing.verify_dataset_integrity import verify_dataset_integrity
from conflunet.utilities.planning_and_configuration import PlansManagerInstanceSeg


def extract_fingerprint_dataset(dataset_id: int,
                                fingerprint_extractor_class: Type[
                                    DatasetFingerprintExtractor] = DatasetFingerprintExtractor,
                                num_processes: int = default_num_processes,
                                clean: bool = True, verbose: bool = True):
    """
    This function is exactly the same as the nnUNet implementation, except for the dataset integrity verification
    """
    fingerprint_extractor = fingerprint_extractor_class(dataset_id, num_processes=num_processes, verbose=verbose)
    return fingerprint_extractor.run(overwrite_existing=clean)


def preprocess_dataset(dataset_id: int,
                       plans_identifier: str = 'nnUNetPlans',
                       configurations: Union[Tuple[str], List[str]] = ('2d', '3d_fullres', '3d_lowres'),
                       num_processes: Union[int, Tuple[int, ...], List[int]] = (8, 4, 8),
                       verbose: bool = False) -> None:
    if not isinstance(num_processes, list):
        num_processes = list(num_processes)
    if len(num_processes) == 1:
        num_processes = num_processes * len(configurations)
    if len(num_processes) != len(configurations):
        raise RuntimeError(
            f'The list provided with num_processes must either have len 1 or as many elements as there are '
            f'configurations (see --help). Number of configurations: {len(configurations)}, length '
            f'of num_processes: '
            f'{len(num_processes)}')

    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(f'Preprocessing dataset {dataset_name}')
    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
    plans_manager = PlansManagerInstanceSeg(plans_file)
    for n, c in zip(num_processes, configurations):
        print(f'Configuration: {c}...')
        if c not in plans_manager.available_configurations:
            print(
                f"INFO: Configuration {c} not found in plans file {plans_identifier + '.json'} of "
                f"dataset {dataset_name}. Skipping.")
            continue
        configuration_manager = plans_manager.get_configuration(c)
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose,
                                                                add_small_object_classes_in_npz = True,
                                                                add_confluent_instances_in_npz = True)
        preprocessor.run(dataset_id, c, plans_identifier, num_processes=n)

    # copy the gt to a folder in the nnUNet_preprocessed so that we can do validation even if the raw data is no
    # longer there (useful for compute cluster where only the preprocessed data is available)
    from distutils.file_util import copy_file
    maybe_mkdir_p(join(nnUNet_preprocessed, dataset_name, 'gt_segmentations'))
    dataset_json = load_json(join(nnUNet_raw, dataset_name, 'dataset.json'))
    dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)
    # only copy files that are newer than the ones already present
    for k in dataset:
        copy_file(dataset[k]['label'],
                  join(nnUNet_preprocessed, dataset_name, 'gt_segmentations', k + dataset_json['file_ending']),
                  update=True)


def preprocess(dataset_id: int,
               check_dataset_integrity: bool = True,
               num_processes: int = default_num_processes,
               overwrite_existing_dataset_fingerprint: bool = False,
               verbose: bool = True):

    # Because the data should already be in the correct format, we can retrieve the dataset.json file provided by the
    # user and verify the dataset integrity.
    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(dataset_name)

    if check_dataset_integrity:
        verify_dataset_integrity(join(nnUNet_raw, dataset_name), num_processes)

    # As in nnUNet, we first have to extract the fingerprint, and plan the experiment. Or at least generate the minimum
    # working files that nnUNet would generate.
    # See x for a detailed description of the expected file strurcture and naming conventions.
    # --> ID.nii.gz in labelsTr is the instance segmentation file.
    # Let's start by extracting the fingerprint from the dataset.json file.
    extract_fingerprint_dataset(dataset_id, num_processes=num_processes, clean=overwrite_existing_dataset_fingerprint,
                                verbose=verbose)

    # The next step in the nnUNet pipeline is to plan the experiment. This produces the necessary files for the
    # preprocessing pipeline.
    plan_experiment_dataset(dataset_id)

    # Make sure the generated files exist
    assert isdir(join(nnUNet_preprocessed, dataset_name)), f"nnUNet_preprocessed/{dataset_name} does not exist"
    assert isfile(join(nnUNet_preprocessed, dataset_name, "dataset.json")), \
        f"nnUNet_preprocessed/{dataset_name}/dataset.json does not exist"
    assert isfile(join(nnUNet_preprocessed, dataset_name, "dataset_fingerprint.json")), \
        f"nnUNet_preprocessed/{dataset_name}/dataset_fingerprint.json does not exist"
    assert isfile(join(nnUNet_preprocessed, dataset_name, "nnUNetPlans.json")), \
        f"nnUNet_preprocessed/{dataset_name}/nnUNetPlans.json does not exist"

    # The preprocessing pipeline is then run in the next step.
    preprocess_dataset(dataset_id, num_processes=(num_processes,), configurations=('3d_fullres',), verbose=verbose)


if __name__=='__main__':
    # folder = "/home/mwynen/data/nnUNet/nnUNet_raw/Dataset321_WMLIS"
    # verify_dataset_integrity(folder)
    preprocess(321, check_dataset_integrity=True, num_processes=8, verbose=True)