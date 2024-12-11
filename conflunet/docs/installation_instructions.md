# System requirements

## Operating system
ConfLUNet has been developed and tested on Linux (Ubuntu 22.04; Red Hat Linux Enterprise 8).

## Hardware requirements
Refer to [nnUNet's hardware requirements](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).

## Installation instructions
First clone this repository:
```commandline
git clone https://github.com/maxencewynen/ConfLUNet/tree/main
cd ConfLUNet
```
As with nnUNet, we strongly recommend that you install ConfLUNet's requirements in a virtual environment! Pip or anaconda are both fine. If you choose to compile PyTorch from source (see below), you will need to use conda instead of pip.

Use a recent version of Python! 3.9 or newer is guaranteed to work!

1. Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website (conda/pip). Please install the latest version with support for your hardware (cuda, mps, cpu). For maximum speed, consider [compiling pytorch yourself](https://github.com/pytorch/pytorch#from-source) (experienced users only!).
2. Install the repo locally 
    ```commandline
    pip install -e
    ```
3. ConfLUNet's file organization is entirely based on nnU-Net's. It needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to set a few environment variables. Please follow nnUNet's instructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).

Installing ConfLUNet will add several new commands to your terminal. These commands are used to run the entire ConfLUNet pipeline. You can execute them from any location on your system. All ConfLUnet commands have the prefix `conflunet_` for easy identification.

Note that these commands simply execute python scripts. If you installed ConfLUNet in a virtual environment, this environment must be activated when executing the commands. You can see what scripts/functions are executed by checking the project scripts in the `pyproject.toml` file.

All ConfLUNet commands have a `-h` (`--help`) option which gives information on how to use them.