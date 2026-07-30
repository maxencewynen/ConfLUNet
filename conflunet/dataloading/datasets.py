import os
from typing import List, Union

import numpy as np
import shutil
import monai

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from nnunetv2.training.dataloading.utils import get_case_identifiers
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from monai.data import CacheDataset, DataLoader
from conflunet.dataloading.transforms import get_train_transforms, get_val_transforms, get_test_transforms


class LesionInstancesDataset(CacheDataset):
    def __init__(self, folder: str,
                 case_identifiers: Union[List[str], None] = None,
                 transforms: monai.transforms = None,
                 cache_rate: float = 1.0,
                 num_workers: int = 0):
        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()

        dataset = []
        for c in case_identifiers:
            dataset.append({'name': c,
                            'data': join(folder, f"{c}.npz"),
                            'properties_file': join(folder, f"{c}.pkl")})

        self.dataset = dataset
        super().__init__(data=dataset, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)

    def make_dataloader(self, batch_size=1, num_workers=0, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


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


def test_lesion_instances_dataset():
    from conflunet.dataloading.utils import save_loaded_data_as_nifti, save_batch_as_nifti
    import torch
    import numpy as np
    import time
    seed_val = 1
    from monai.utils import set_determinism
    set_determinism(seed=seed_val)

    # test training transforms first
    from monai.transforms import Compose, RandSpatialCropd
    from conflunet.dataloading.transforms import CustomLoadNPZInstanced, DeleteKeysd
    import torch.utils.data.sampler as sampler
    ds = LesionInstancesDataset('/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans_3d_fullres',
                                transforms=Compose([
                                    CustomLoadNPZInstanced(keys=['data']),
                                    RandSpatialCropd(keys=['img', 'seg', 'instance_seg', 'brainmask'],
                                                     roi_size=(16,16,16),
                                                     random_center=False, random_size=False),
                                    DeleteKeysd(keys=['properties']),]), cache_rate=0)

    # dl = ds.make_dataloader(batch_size=1, num_workers=0, shuffle=True)
    dl = DataLoader(ds, batch_size=2, num_workers=0,
                    sampler=sampler.RandomSampler(ds, replacement=True, num_samples=2))

    def infinite_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    dli = infinite_dataloader(dl)
    for i in range(10):
        print(i, next(dli)['name'])



    ds = LesionInstancesDataset('/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans_3d_fullres',
                                transforms=get_train_transforms(seed=1), cache_rate=0)
    # all_times = []
    # for i in range(5):
    #     start = time.time()
    #     a = ds[0]
    #     elapsed = time.time() - start
    #     print("Time to load batch: ", elapsed)
    #     all_times.append(elapsed)
    # print("Average time to load batch: ", np.mean(all_times))
    # print("Length of dataset: ", len(ds))
    # print("Length of dataset[0]:", len(a))

    # dl = get_val_dataloader('/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans_3d_fullres')
    # for epoch in range(3):
    #     for i, batch in enumerate(dl):
    #         if i > 0: continue
    #         # save the first batch to nifti
    #         save_batch_as_nifti(batch, output_folder=f'/home/mwynen/data/nnUNet/tmp/epoch_{epoch}')
    # save_loaded_data_as_nifti(ds[0], output_folder='/home/mwynen/data/nnUNet/tmp', n_patches_to_save=1)


    # dl = get_test_dataloader('/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans_3d_fullres')
    # for i, batch in enumerate(dl):
    #     if i > 0: continue
    #     save_batch_as_nifti(batch, output_folder='/home/mwynen/data/nnUNet/tmp', n_patches_to_save=1)

    # for i, data in enumerate(ds[0]):
    #     key = "instance_seg"
    #
    #     img = data[key]
    #     if isinstance(img, torch.Tensor):
    #         img = img.cpu().numpy()
    #
    #     img = np.squeeze(img)
    #
    #     print(i, np.unique(img))


class nnUNetInstancesDataset(nnUNetDataset):
    # adapted from nnunetv2.training.dataloading.nnunet_dataset to handle instance segmentation maps
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):
        super().__init__(folder, case_identifiers, num_images_properties_loading_threshold,
                         folder_with_segs_from_previous_stage)

    def load_case(self, key):
        data, seg, _ = super().load_case(key)
        entry = self[key]

        if 'open_instance_seg_file' in entry.keys():
            instance_seg = entry['open_instance_seg_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + "_instance_seg.npy"):
            instance_seg = np.load(entry['data_file'][:-4] + "_instance_seg.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_instance_seg_file'] = instance_seg
                # print('saving open seg file')
        else:
            instance_seg = np.load(entry['data_file'])['instance_seg']

        if 'seg_from_prev_stage_file' in entry.keys():
            if isfile(entry['seg_from_prev_stage_file'][:-4] + ".npy"):
                instance_seg_prev = np.load(entry['instance_seg_from_prev_stage_file'][:-4] + ".npy", 'r')
            else:
                instance_seg_prev = np.load(entry['instance_seg_from_prev_stage_file'])['instance_seg']
            seg = np.vstack((instance_seg, instance_seg_prev[None]))

        return data, seg, instance_seg, entry['properties']


if __name__=='__main__':
    test_lesion_instances_dataset()

