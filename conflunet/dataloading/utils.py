import os

from conflunet.dataloading.datasets import LesionInstancesDataset
import torch
import numpy as np
from os.path import exists, join

from conflunet.preprocessing.utils import save_nii


def save_loaded_data_as_nifti(batch_data, n_patches_to_save=2, output_folder='./tmp', postfix=''):
    if not exists(output_folder):
        os.makedirs(output_folder)
    for i, data in enumerate(batch_data):
        if i >= n_patches_to_save:
            break
        for key in ['img', 'instance_seg', 'seg', 'foreground', 'center_heatmap', 'offsets', 'label']:
            if key not in data.keys():
                continue

            img = data[key]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()

            img = np.squeeze(img)

            if len(img.shape) == 4:
                for j in range(img.shape[0]):
                    save_nii(join(f"{output_folder}", f"{data['name']}_{i}_{key}_{j}{postfix}.nii.gz"), img[j])
            else:
                save_nii(join(f"{output_folder}", f"{data['name']}_{i}_{key}{postfix}.nii.gz"), img)


def save_batch_as_nifti(batch_data, output_folder='./tmp', postfix='', n_patches_to_save=1):
    if not exists(output_folder):
        os.makedirs(output_folder)
    for key in ['img', 'instance_seg', 'seg', 'foreground', 'center_heatmap', 'offsets', 'label']:
        if key not in batch_data.keys():
            continue

        img = batch_data[key]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        for i in range(n_patches_to_save):
            if img.shape[1] == 1:
                img_ = np.squeeze(img)
                save_nii(join(f"{output_folder}", f"{batch_data['name'][i]}_{i}_{key}{postfix}.nii.gz"), img_[i])
            else:
                for j in range(img.shape[1]):
                    save_nii(join(f"{output_folder}", f"{batch_data['name'][i]}_{i}_{key}_{j}{postfix}.nii.gz"), img[i][j])


