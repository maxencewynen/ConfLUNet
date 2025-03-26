import os
import shutil
from shutil import copy, copytree
from os.path import join as pjoin

from conflunet.utilities.planning_and_configuration import load_dataset_and_configuration

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results


def compress_model(dataset_id, model_name, output_dir):
    os.makedirs(pjoin(output_dir, 'tmp'), exist_ok=False)

    dataset_name, _, _, _ = load_dataset_and_configuration(int(dataset_id))

    os.makedirs(pjoin(output_dir, 'tmp', 'nnUNet_results', dataset_name), exist_ok=False)
    os.makedirs(pjoin(output_dir, 'tmp', 'nnUNet_preprocessed', dataset_name), exist_ok=False)
    os.makedirs(pjoin(output_dir, 'tmp', 'nnUNet_raw', dataset_name), exist_ok=False)

    copy(pjoin(nnUNet_preprocessed, dataset_name, 'nnUNetPlans.json'),
         pjoin(output_dir, 'tmp', 'nnUNet_preprocessed', dataset_name))
    copy(pjoin(nnUNet_preprocessed, dataset_name, 'dataset.json'),
         pjoin(output_dir, 'tmp', 'nnUNet_preprocessed', dataset_name))
    copy(pjoin(nnUNet_preprocessed, dataset_name, 'dataset.json'),
         pjoin(output_dir, 'tmp', 'nnUNet_raw', dataset_name))

    copytree(pjoin(nnUNet_results, dataset_name, model_name),
         pjoin(output_dir, 'tmp', 'nnUNet_results', dataset_name, model_name), dirs_exist_ok=True)

    shutil.make_archive(pjoin(output_dir, model_name + '_EXPORT'), 'zip', pjoin(output_dir, 'tmp'))
    shutil.rmtree(pjoin(output_dir, 'tmp'))

    print(f"Saved compressed model at {pjoin(output_dir, model_name + '_EXPORT')}.zip .")


def export_pretrained_model_entry():
    import argparse
    parser = argparse.ArgumentParser(
        description='Export all necessary info to run inference on a pretrained model')
    parser.add_argument('--dataset_id', type=str, help='Dataset ID')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--output_dir', type=str, help='Path to where to store the compressed file')
    args = parser.parse_args()

    compress_model(args.dataset_id, args.model_name, args.output_dir)


if __name__ == '__main__':
    export_pretrained_model_entry()

