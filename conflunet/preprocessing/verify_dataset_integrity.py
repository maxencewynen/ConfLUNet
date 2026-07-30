from nnunetv2.experiment_planning.verify_dataset_integrity import check_cases
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
import multiprocessing
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json, isdir, subfiles


def verify_dataset_integrity(folder: str, num_processes: int = 8) -> None:
    """
    This function is used to verify the integrity of the dataset. It is used to check if the dataset is in the correct format.
    This differs only slightly from the nnUNet implementation, as we do not have the same segmentation files.
    The only difference is that we do not use nnunetv2.experiment_planning.verify_dataset_integrity.verify_labels

    folder needs the imagesTr, imagesTs and labelsTr subfolders. There also needs to be a dataset.json
    checks if the expected number of training cases and labels are present
    for each case, if possible, checks whether the pixel grids are aligned
    checks whether the labels really only contain values they should

    :param folder:
    :param num_processes:
    :return:
    """
    assert isfile(join(folder, "dataset.json")), f"There needs to be a dataset.json file in folder, folder={folder}"
    dataset_json = load_json(join(folder, "dataset.json"))

    if not 'dataset' in dataset_json.keys():
        assert isdir(join(folder, "imagesTr")), f"There needs to be a imagesTr subfolder in folder, folder={folder}"
        assert isdir(join(folder, "labelsTr")), f"There needs to be a labelsTr subfolder in folder, folder={folder}"

    # make sure all required keys are there
    dataset_keys = list(dataset_json.keys())
    required_keys = ["channel_names", "numTraining", "file_ending"]
    assert all([i in dataset_keys for i in required_keys]), 'not all required keys are present in dataset.json.' \
                                                            '\n\nRequired: \n%s\n\nPresent: \n%s\n\nMissing: ' \
                                                            '\n%s\n\nUnused by nnU-Net:\n%s' % \
                                                            (str(required_keys),
                                                             str(dataset_keys),
                                                             str([i for i in required_keys if i not in dataset_keys]),
                                                             str([i for i in dataset_keys if i not in required_keys]))

    expected_num_training = dataset_json['numTraining']
    num_modalities = len(dataset_json['channel_names'].keys()
                         if 'channel_names' in dataset_json.keys()
                         else dataset_json['modality'].keys())
    file_ending = dataset_json['file_ending']

    dataset = get_filenames_of_train_images_and_targets(folder, dataset_json)

    # check if the right number of training cases is present
    assert len(dataset) == expected_num_training, 'Did not find the expected number of training cases ' \
                                                               '(%d). Found %d instead.\nExamples: %s' % \
                                                               (expected_num_training, len(dataset),
                                                                list(dataset.keys())[:5])

    # check if corresponding labels are present
    if 'dataset' in dataset_json.keys():
        # just check if everything is there
        ok = True
        missing_images = []
        missing_labels = []
        for k in dataset:
            for i in dataset[k]['images']:
                if not isfile(i):
                    missing_images.append(i)
                    ok = False
            if not isfile(dataset[k]['label']):
                missing_labels.append(dataset[k]['label'])
                ok = False
        if not ok:
            raise FileNotFoundError(f"Some expected files were missing. Make sure you are properly referencing them "
                                    f"in the dataset.json. Or use imagesTr & labelsTr folders!\nMissing images:"
                                    f"\n{missing_images}\n\nMissing labels:\n{missing_labels}")
    else:
        # old code that uses imagestr and labelstr folders
        labelfiles = subfiles(join(folder, 'labelsTr'), suffix=file_ending, join=False)
        label_identifiers = [i[:-len(file_ending)] for i in labelfiles]
        labels_present = [i in label_identifiers for i in dataset.keys()]
        missing = [i for j, i in enumerate(dataset.keys()) if not labels_present[j]]
        assert all(labels_present), f'not all training cases have a label file in labelsTr. Fix that. Missing: {missing}'

    reader_writer_class = determine_reader_writer_from_dataset_json(dataset_json,
                                                                    dataset[dataset.keys().__iter__().__next__()][
                                                                        'images'][0])

    label_files = [v['label'] for v in dataset.values()]
    image_files = [v['images'] for v in dataset.values()]

    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        # check whether shapes and spacings match between images and labels
        result = p.starmap(
            check_cases,
            zip(image_files, label_files, [num_modalities] * expected_num_training,
                [reader_writer_class] * expected_num_training)
        )
        if not all(result):
            raise RuntimeError(
                'Some images have errors. Please check text output above to see which one(s) and what\'s going on.')

    # check for nans
    # check all same orientation nibabel
    print('\n####################')
    print('verify_dataset_integrity Done. \nIf you didn\'t see any error messages then your dataset is most likely OK!')
    print('####################\n')