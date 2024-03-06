# ConfLUNet: Improving Confluent Lesion Identification in Multiple Sclerosis with Instance Segmentation

This repository contains the code necessary to reproduce the results presented in the paper published at IEEE: ISBI titled "ConfLUNet: Improving Confluent Lesion Identification in Multiple Sclerosis with Instance Segmentation".

## Goal

In multiple sclerosis (MS), white matter lesion (WML) instance masks are very relevant to enhance diagnosis and disease monitoring. Yet, all existing automated WML segmentation methods aim at improving a semantic segmentation model and postprocessing it to group voxels together into lesion instances. In this paper, we propose ConfLUNet - the first end-to-end instance segmentation model designed to detect and segment WML instances in MS.

## Contents

* conflunet/
  * data_load.py: Utility file for data handling, including building Monai datasets and setting up transforms.
  * inference.py: Script for inferring a trained model on data and postprocessing the results.
  * metrics.py: Implementation of evaluation metrics used in the paper.
  * model.py: PyTorch implementations of the models' architectures.
  * postprocess.py: Utility functions for postprocessing model predictions into instance segmentations.
  * train.py: Script for training the models with specified parameters.
  * unit_tests.py: Unit tests for the project.
  * utils.py: Utility functions.
* README.md: This file.
* requirements.txt: List of Python library requirements for the repository to work correctly.

## Requirements

* Python >= 3.10
* PyTorch
* GPU with at least 12GB VRAM
* Other libraries listed in requirements.txt

## Usage

1. Set up two environment variables:
   * `DATA_ROOT_DIR`: Path to the data files.
   * `MODELS_ROOT_DIR`: Path to the trained models.
2. Organize data as follows:
```
 /path/to/data_root_dir/
├── all
│   ├── brainmasks
│   │   └── sub-001_brainmask.nii.gz
│   │   └── sub-002_brainmask.nii.gz
│   │   └── sub-003_brainmask.nii.gz
│   ├── images
│   │   └── sub-001_FLAIR.nii.gz
│   │   └── sub-002_FLAIR.nii.gz
│   │   └── sub-003_FLAIR.nii.gz
│   ├── labels
│   │   └── sub-001_mask-instances.nii.gz
│   │   └── sub-002_mask-instances.nii.gz
│   │   └── sub-003_mask-instances.nii.gz
├── predictions
│   ├── test
│   │   └── trained_model1
│   └── val
│       └── trained_model1
├── test
│   ├── brainmasks
│   │   └── sub-001_brainmask.nii.gz
│   ├── images
│   │   └── sub-001_FLAIR.nii.gz
│   └── labels
│       └── sub-001_mask-instances.nii.gz
├── train
│   ├── brainmasks
│   │   └── sub-002_brainmask.nii.gz
│   ├── images
│   │   └── sub-002_FLAIR.nii.gz
│   └── labels
│       └── sub-002_mask-instances.nii.gz
└── val
    ├── brainmasks
    │   └── sub-003_brainmask.nii.gz
    ├── images
    │   └── sub-003_FLAIR.nii.gz
    └── labels
        └── sub-003_mask-instances.nii.gz
```

## How to run

* Ensure environment variables are set.
* Run `train.py` to train the models.
* Run `inference.py` to infer a trained model on data.

## Contributing

Feel free to open an issue or pull request if you have any suggestions, questions, or improvements.

## License

This project is licensed under the MIT LICENSE - see the LICENSE file for details.



