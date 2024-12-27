import os
from os.path import exists as pexists
import monai
from typing import List, Union, Tuple

from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.utils import get_case_identifiers

from conflunet.dataloading.datasets import LesionInstancesDataset
from conflunet.utilities.planning_and_configuration import load_dataset_and_configuration
from conflunet.dataloading.utils import get_train_transforms, get_val_transforms, get_test_transforms


def get_train_dataloader(folder: str,
                         case_identifiers: Union[List[str], None] = None,
                         patch_size: Union[tuple, None] = (96, 96, 96),
                         batch_size=2,
                         remove_small_instances=False,
                         voxel_size=(1, 1, 1),
                         minimum_instance_size=14,
                         minimum_size_along_axis=3,
                         num_workers=0,
                         cache_rate=1.0,
                         seed_val=1,
                         get_small_instances=False,
                         get_confluent_instances=False) -> monai.data.DataLoader:
    if pexists(join(os.path.abspath(os.path.join(folder, os.pardir)),  'shapes')):
        path_to_shapes_json = join(os.path.abspath(os.path.join(folder, os.pardir)),  'shapes', 'shapes.json')
    else:
        path_to_shapes_json = None
    train_transforms = get_train_transforms(seed=seed_val, patch_size=patch_size,
                                            remove_small_instances=remove_small_instances,
                                            voxel_size=voxel_size,
                                            minimum_instance_size=minimum_instance_size,
                                            minimum_size_along_axis=minimum_size_along_axis,
                                            get_small_instances=get_small_instances,
                                            get_confluent_instances=get_confluent_instances,
                                            path_to_shapes_json=path_to_shapes_json)
    ds = LesionInstancesDataset(folder, case_identifiers,
                                transforms=train_transforms,
                                cache_rate=cache_rate)
    return ds.make_dataloader(batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_val_dataloader(folder: str,
                       case_identifiers: Union[List[str], None] = None,
                       patch_size: Union[tuple, None] = (96, 96, 96),
                       batch_size=2,
                       remove_small_instances=False,
                       voxel_size=(1, 1, 1),
                       minimum_instance_size=14,
                       minimum_size_along_axis=3,
                       num_workers=0,
                       cache_rate=1.0,
                       seed_val=1) -> monai.data.DataLoader:
    val_transforms = get_train_transforms(seed=seed_val, patch_size=patch_size,
                                          remove_small_instances=remove_small_instances,
                                          voxel_size=voxel_size,
                                          minimum_instance_size=minimum_instance_size,
                                          minimum_size_along_axis=minimum_size_along_axis)
    ds = LesionInstancesDataset(folder, case_identifiers,
                                transforms=val_transforms,
                                cache_rate=cache_rate)
    return ds.make_dataloader(batch_size=batch_size, num_workers=num_workers, shuffle=False)


def get_test_dataloader(folder: str,
                        case_identifiers: Union[List[str], None] = None,
                        batch_size=1,
                        voxel_size=(1, 1, 1),
                        minimum_instance_size=14,
                        minimum_size_along_axis=3,
                        num_workers=0,
                        test=True) -> monai.data.DataLoader:
    test_transforms = get_test_transforms(test=test,
                                          voxel_size=voxel_size,
                                          minimum_instance_size=minimum_instance_size,
                                          minimum_size_along_axis=minimum_size_along_axis)
    ds = LesionInstancesDataset(folder, case_identifiers, transforms=test_transforms, cache_rate=0.0)
    return ds.make_dataloader(batch_size=batch_size, num_workers=num_workers, shuffle=False)


def _get_val_train_keys(preprocessed_dataset_folder: str, fold: int = None) -> Tuple[List[str], List[str]]:
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
        num_workers: int =0,
        cache_rate: float =1.0,
        seed_val: int =1,
        get_small_instances=False,
        get_confluent_instances=False) -> monai.data.DataLoader:
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(dataset_id)
    # TODO: handle case when dataset is not preprocessed
    preprocessed_dataset_folder = join(nnUNet_preprocessed, dataset_name)
    preprocessed_data_folder = join(preprocessed_dataset_folder, configuration.configuration['data_identifier'])

    tr_keys, val_keys = _get_val_train_keys(preprocessed_dataset_folder, fold)

    patch_size = configuration.patch_size
    batch_size = configuration.batch_size

    return get_train_dataloader(preprocessed_data_folder,
                                case_identifiers=tr_keys, patch_size=patch_size, batch_size=batch_size,
                                num_workers=num_workers, cache_rate=cache_rate, seed_val=seed_val,
                                get_small_instances=get_small_instances, get_confluent_instances=get_confluent_instances)


def get_val_dataloader_from_dataset_id_and_fold(
        dataset_id: Union[int, str],
        fold: int = None,
        num_workers: int = 0,
        cache_rate: float = 1.0,
        seed_val: int = 1) -> monai.data.DataLoader:
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(dataset_id)
    # TODO: handle case when dataset is not preprocessed
    preprocessed_dataset_folder = join(nnUNet_preprocessed, dataset_name)
    preprocessed_data_folder = join(preprocessed_dataset_folder, configuration.configuration['data_identifier'])

    tr_keys, val_keys = _get_val_train_keys(preprocessed_dataset_folder, fold)

    patch_size = configuration.patch_size
    batch_size = configuration.batch_size

    return get_val_dataloader(preprocessed_data_folder,
                              case_identifiers=val_keys, patch_size=patch_size, batch_size=batch_size,
                              num_workers=num_workers, cache_rate=cache_rate, seed_val=seed_val)


def get_full_val_dataloader_from_dataset_id_and_fold(
        dataset_id: Union[int, str],
        fold: int = None,
        num_workers: int = 0) -> monai.data.DataLoader:
    """Retrieves full images instead of patches"""
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(dataset_id)
    # TODO: handle case when dataset is not preprocessed
    preprocessed_dataset_folder = join(nnUNet_preprocessed, dataset_name)
    preprocessed_data_folder = join(preprocessed_dataset_folder, configuration.configuration['data_identifier'])

    tr_keys, val_keys = _get_val_train_keys(preprocessed_dataset_folder, fold)

    return get_test_dataloader(preprocessed_data_folder, case_identifiers=val_keys, batch_size=1,
                               num_workers=num_workers, test=False, voxel_size=configuration.spacing,
                               minimum_instance_size=14, minimum_size_along_axis=3)


if __name__=="__main__":
    import numpy as np
    import nibabel as nib
    import torch
    import random
    import time
    seed_val = 1

    # seeding
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    save_dir = r"/home/mwynen/data/nnUNet/tmp/copy_paste"
    dataset_id = 322 #321
    fold = 0 #2
    num_workers = 0
    cache_rate = 0
    train_loader = get_train_dataloader_from_dataset_id_and_fold(dataset_id, fold, num_workers, cache_rate, seed_val)
    print(train_loader)
    for epoch in range(6):
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            print()
            print(f"---- Epoch {epoch}, batch {i}, time: {time.time()-t0} ----")
            print(i, batch['img'].shape)
            print()
            for j in range(batch['img'].shape[0]):
                # save
                data = np.squeeze(batch['img'][j,0,:,:,:].numpy())
                data2 = np.squeeze(batch['img'][j,1,:,:,:].numpy())
                seg = np.squeeze(batch['seg'][j,0,:,:,:].numpy())
                instance_seg = np.squeeze(batch['instance_seg'][j,0,:,:,:].numpy())
                offsets_x = np.squeeze(batch['offsets'][j,0,:,:,:].numpy())
                offsets_y = np.squeeze(batch['offsets'][j,1,:,:,:].numpy())
                offsets_z = np.squeeze(batch['offsets'][j,2,:,:,:].numpy())
                nawm = np.squeeze(batch['nawm'][j,0,:,:,:].numpy())
                brainmask = np.squeeze(batch['brainmask'][j,0,:,:,:].numpy())

                nib.save(nib.Nifti1Image(data, np.eye(4)), join(save_dir, f"epoch_{epoch}_img_{i}_{j}.nii.gz"))
                nib.save(nib.Nifti1Image(data2, np.eye(4)), join(save_dir, f"epoch_{epoch}_img2_{i}_{j}.nii.gz"))
                nib.save(nib.Nifti1Image(seg, np.eye(4)), join(save_dir, f"epoch_{epoch}_seg_{i}_{j}.nii.gz"))
                nib.save(nib.Nifti1Image(instance_seg, np.eye(4)), join(save_dir, f"epoch_{epoch}_instance_seg_{i}_{j}.nii.gz"))
                nib.save(nib.Nifti1Image(offsets_x, np.eye(4)), join(save_dir, f"epoch_{epoch}_offsets_x_{i}_{j}.nii.gz"))
                nib.save(nib.Nifti1Image(offsets_y, np.eye(4)), join(save_dir, f"epoch_{epoch}_offsets_y_{i}_{j}.nii.gz"))
                nib.save(nib.Nifti1Image(offsets_z, np.eye(4)), join(save_dir, f"epoch_{epoch}_offsets_z_{i}_{j}.nii.gz"))
                nib.save(nib.Nifti1Image(nawm, np.eye(4)), join(save_dir, f"epoch_{epoch}_nawm_{i}_{j}.nii.gz"))
                nib.save(nib.Nifti1Image(brainmask, np.eye(4)), join(save_dir, f"epoch_{epoch}_brainmask_{i}_{j}.nii.gz"))

            if i > 4:
                break
    print("Done!")
