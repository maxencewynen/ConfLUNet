from typing import List, Union

from nnunetv2.training.dataloading.utils import get_case_identifiers
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json, isdir, subfiles, maybe_mkdir_p
from nnunetv2.paths import nnUNet_preprocessed

from conflunet.dataloading.utils import get_train_transforms, get_val_transforms, get_test_transforms
from conflunet.dataloading.datasets import LesionInstancesDataset
from conflunet.preprocessing.preprocess import PlansManagerInstanceSeg


def get_train_dataloader(folder: str,
                         case_identifiers: Union[List[str], None] = None,
                         patch_size: Union[tuple, None] = (96, 96, 96),
                         batch_size=2,
                         num_workers=0,
                         cache_rate=1.0,
                         seed_val=1):
    ds = LesionInstancesDataset(folder, case_identifiers,
                                transforms=get_train_transforms(seed=seed_val, patch_size=patch_size),
                                cache_rate=cache_rate)
    return ds.make_dataloader(batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_val_dataloader(folder: str,
                       case_identifiers: Union[List[str], None] = None,
                       patch_size: Union[tuple, None] = (96, 96, 96),
                       batch_size=2,
                       num_workers=0,
                       cache_rate=1.0,
                       seed_val=1):
    ds = LesionInstancesDataset(folder, case_identifiers,
                                transforms=get_val_transforms(seed=seed_val, patch_size=patch_size),
                                cache_rate=cache_rate)
    return ds.make_dataloader(batch_size=batch_size, num_workers=num_workers, shuffle=False)


def get_test_dataloader(folder: str,
                        case_identifiers: Union[List[str], None] = None,
                        batch_size=1,
                        num_workers=0):
    ds = LesionInstancesDataset(folder, case_identifiers, transforms=get_test_transforms(), cache_rate=0.0)
    return ds.make_dataloader(batch_size=batch_size, num_workers=num_workers, shuffle=False)


def _get_val_train_keys(preprocessed_dataset_folder: str, fold: int = None):
    "Return the train and validation keys for a given fold"
    if fold == 'all' or fold is None:
        case_identifiers = get_case_identifiers(preprocessed_dataset_folder)
        tr_keys = case_identifiers
        val_keys = tr_keys
    else:
        splits_final = join(preprocessed_dataset_folder, 'splits_final.json')
        if not isfile(splits_final):
            raise NotImplementedError(
                f"Could not find splits_final.json for dataset {preprocessed_dataset_folder}. "
                f"In the future, this will be generated automatically but for now you have to do write it "
                f"manually. See the nnUNet documentation for more information on how to do this."
            )
            # TODO: handle case where splits_final.json does not exist
        splits_final = load_json(splits_final)
        tr_keys = splits_final[fold]['train']
        val_keys = splits_final[fold]['val']

    return tr_keys, val_keys


def get_train_dataloader_from_dataset_id_and_fold(
        dataset_id: Union[int, str],
        fold: int = None,
        num_workers=0,
        cache_rate=1.0,
        seed_val=1):
    dataset_name = convert_id_to_dataset_name(dataset_id)
    plans_file = join(nnUNet_preprocessed, dataset_name, 'nnUNetPlans.json')
    plans_manager = PlansManagerInstanceSeg(plans_file)
    configuration = plans_manager.get_configuration('3d_fullres')
    dataset_json = load_json(join(nnUNet_preprocessed, dataset_name, 'dataset.json'))
    # TODO: handle case when dataset is not preprocessed
    preprocessed_dataset_folder = join(nnUNet_preprocessed, dataset_name)
    preprocessed_data_folder = join(preprocessed_dataset_folder, configuration.configuration['data_identifier'])

    tr_keys, val_keys = _get_val_train_keys(preprocessed_dataset_folder, fold)

    patch_size = (20,20,20)#configuration.patch_size
    batch_size = 3#configuration.batch_size

    return get_train_dataloader(preprocessed_data_folder,
                                case_identifiers=tr_keys, patch_size=patch_size, batch_size=batch_size,
                                num_workers=num_workers, cache_rate=cache_rate, seed_val=seed_val)


if __name__=="__main__":
    dataset_id = 321
    fold = 0
    num_workers = 0
    cache_rate = 0
    seed_val = 1
    train_loader = get_train_dataloader_from_dataset_id_and_fold(dataset_id, fold, num_workers, cache_rate, seed_val)
    print(train_loader)
    for i, batch in enumerate(train_loader):
        print(i, batch['img'].shape)
        break
    print("Done!")
