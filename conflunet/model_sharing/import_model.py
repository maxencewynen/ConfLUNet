import os
import shutil
from shutil import copy, copytree
from os.path import join as pjoin
from os.path import exists as pexists

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results, nnUNet_raw


def import_model(compressed_file):
    assert compressed_file.endswith('.zip'), f"Expected export file to be a .zip file but got .{compressed_file.split('.')[-1]}"
    dirname = os.path.dirname(compressed_file)
    shutil.unpack_archive(compressed_file, dirname, 'zip')

    dataset_name = None
    for i, dataset_name in enumerate(os.listdir(pjoin(dirname, 'nnUNet_results'))):
        if i > 0:
            raise Exception

    model_name = os.listdir(pjoin(dirname, 'nnUNet_results', dataset_name))[0]
    if pexists(pjoin(nnUNet_results, dataset_name, model_name)):
        raise FileExistsError(f"Model {model_name} for dataset {dataset_name} already exists "
                              f"({pjoin(nnUNet_results, dataset_name, model_name)})")

    shutil.move(pjoin(dirname, 'nnUNet_results', dataset_name), nnUNet_results)
    shutil.move(pjoin(dirname, 'nnUNet_preprocessed', dataset_name), nnUNet_preprocessed)
    shutil.move(pjoin(dirname, 'nnUNet_raw', dataset_name), nnUNet_raw)

    print(f"Succesful import of model {model_name} for dataset {dataset_name}.")


def import_pretrained_model_entry():
    import argparse
    parser = argparse.ArgumentParser(
        description='Import pretrained model from compressed zip file')
    parser.add_argument('input_file', type=str, help='compressed zip file (obtained by using the `export_pretrained_model_entry` command)')
    args = parser.parse_args()

    import_model(args.input_file)


if __name__ == '__main__':
    import_pretrained_model_entry()

