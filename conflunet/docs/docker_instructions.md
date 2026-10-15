# Using ConfLUNet with Docker

This guide explains how to use ConfLUNet through Docker (or Singularity/Apptainer) containers. Containers provide a consistent environment, ensuring dependencies are configured correctly regardless of your host system.

Starting with version `0.3.0`, the image ships with a **default pre-trained model (Dataset321_WMLIS, FLAIR)** baked in, so inference works out of the box with no model download or mount.

## System Requirements

- Docker (or Singularity/Apptainer) installed
- For GPU acceleration: NVIDIA Container runtime and a CUDA-capable GPU with drivers
- CPU-only execution is supported (inference falls back to CPU automatically; it is slow, and training on CPU is not practical)
- Disk space for the image (~5.6 GB, includes the bundled model) and your data
- Adequate shared memory (the suggested 40GB is approximate; adjust to dataset size)

## Getting the Image

### Pre-built image (Docker Hub)
```bash
docker pull petermcgor/conflunet:0.3.0
```

### Build from source
From the repository root (the build bundles the model from `conflunet_model/`; see that folder's README to populate the weights before building):
```bash
docker build -t conflunet .
```

## A Note on the nnU-Net Directory Structure

ConfLUNet follows nnU-Net's conventions. The nnU-Net framework organises data into three top-level directories:

```
/path/to/your/data/
├── nnUNet_raw/            # Raw data: DatasetXXX_Name/ with images (*_0000.*) and dataset.json
├── nnUNet_preprocessed/   # Populated by preprocessing
└── nnUNet_results/        # Populated by training (your trained models)
```

Inside the container these three locations are read from the `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` environment variables, which point under `/opt/conflunet_resources` by default.

- **Training and preprocessing require this structure.**
- **Inference with the bundled model does not.**

> ## ⚡ **Inference with the bundled model does NOT require the nnU-Net structure.**
> **Just point `--input_dir` at a folder of FLAIR images named `*_0000.*`. See [Quick Start](#quick-start-inference-with-the-bundled-model) below.**

If you already work with nnU-Net, you likely have this layout already and can mount it directly. For the full specification of the raw dataset format and the inference input format, see the official nnU-Net documentation: [dataset_format.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) and [dataset_format_inference.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format_inference.md).

## Quick Start: Inference with the Bundled Model

The default model is baked in, so you only provide input images and an output directory. No model files and no `nnUNet_*` structure are required.

Input requirement: `--input_dir` is simply a folder containing FLAIR image(s) named with nnU-Net's channel suffix, e.g. `sub-001_0000.nii.gz` (single channel `_0000` for FLAIR). Nothing else needs to be in that folder. A folder from an existing nnU-Net dataset (e.g. an `imagesTs` directory) works too, since it is also just a folder of `*_0000` images. Files without the `_0000` suffix are not detected; if your files use other names (e.g. BIDS), rename or symlink them to add `_0000`.

```bash
docker run -it --gpus all --shm-size=40gb \
  -v /path/to/your/flair_inputs:/input \
  -v /path/to/your/outputs:/output \
  petermcgor/conflunet:0.3.0 \
  conflunet_predict --dataset_id 321 \
    --input_dir /input --output_dir /output \
    --model ConfLUNet_FLAIR_v24-12-11 --postprocessor ConfLUNet
```

`--dataset_id 321` selects the bundled model. For CPU-only, omit `--gpus all`.

> **Note:** `--dataset_id 321` always uses the model baked into the image. If you happen to have your own `Dataset321_WMLIS` in a mounted `nnUNet_results`, it is **ignored** for inference; the bundled model takes precedence. To run your own model, give it a different dataset id (see [Inference with your own model](#inference-with-your-own-model)).

## Using Your Own Models / Training

For training, or inference with your own nnU-Net/ConfLUNet-trained models, work within the standard nnU-Net framework and mount a data directory with its structure to `/opt/conflunet_resources`:

```
/path/to/your/data/
├── nnUNet_raw/            # DatasetXXX_Name/ with images (*_0000.*) and dataset.json
├── nnUNet_preprocessed/   # Populated by preprocessing
└── nnUNet_results/        # Populated by training
```

Typical flow: place raw data under `nnUNet_raw/DatasetXXX_Name/` with its `dataset.json`; run preprocessing to populate `nnUNet_preprocessed/`; run training to populate `nnUNet_results/`. For the exact raw format, see the nnU-Net [dataset_format.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).

```bash
docker run -it --gpus all --shm-size=40gb \
  -v /path/to/your/data:/opt/conflunet_resources \
  petermcgor/conflunet:0.3.0 [command]
```

The bundled Dataset321 model lives in a separate path (`/opt/conflunet_default_inference_model`), so mounting your own data over `/opt/conflunet_resources` does not hide it; `--dataset_id 321` still works alongside your own datasets (and always resolves to the bundled model, as noted above).

### Preprocessing
```bash
docker run -it --gpus all --shm-size=40gb \
  -v /path/to/your/data:/opt/conflunet_resources \
  petermcgor/conflunet:0.3.0 \
  conflunet_plan_and_preprocess --dataset_id [ID] --check_dataset_integrity
```

### Training
```bash
docker run -it --gpus all --shm-size=40gb \
  -v /path/to/your/data:/opt/conflunet_resources \
  petermcgor/conflunet:0.3.0 \
  conflunet_train --dataset_id [ID] --fold [FOLD] --model_name [NAME]
```
Disable Weights & Biases logging by adding `--wandb_ignore`.

### Inference with your own model
Same as the Quick Start, but with `--dataset_id` set to your own dataset (not 321) and your data mounted at `/opt/conflunet_resources`, so the model is read from your mounted `nnUNet_results`:
```bash
docker run -it --gpus all --shm-size=40gb \
  -v /path/to/your/data:/opt/conflunet_resources \
  petermcgor/conflunet:0.3.0 \
  conflunet_predict --dataset_id [ID] \
    --input_dir /path/to/inputs --output_dir /path/to/outputs \
    --model [YOUR_MODEL_NAME] --postprocessor ConfLUNet
```

### Interactive shell
```bash
docker run -it --gpus all --shm-size=40gb \
  -v /path/to/your/data:/opt/conflunet_resources \
  petermcgor/conflunet:0.3.0 bash
```

## Running with Singularity / Apptainer

The same image runs under Singularity/Apptainer, common on HPC clusters without Docker. Convert once:
```bash
apptainer pull conflunet_0.3.0.sif docker://petermcgor/conflunet:0.3.0
```

Inference with the bundled model (`--nv` for GPU; `--bind` any paths outside your home):
```bash
apptainer exec --nv \
  --bind /path/to/inputs:/input --bind /path/to/outputs:/output \
  conflunet_0.3.0.sif \
  conflunet_predict --dataset_id 321 \
    --input_dir /input --output_dir /output \
    --model ConfLUNet_FLAIR_v24-12-11 --postprocessor ConfLUNet
```

Notes:
- Apptainer auto-mounts your home and (usually) scratch; other filesystems must be bound with `--bind source:dest`.
- The image is read-only under Apptainer. For training, point the `nnUNet_*` locations at writable, mounted directories.
- Request a GPU through your scheduler (e.g. SLURM `--gres=gpu:1`); `--nv` only exposes GPUs the job already has.

## Notes

- Bundled model: `Dataset321_WMLIS` / `ConfLUNet_FLAIR_v24-12-11` (FLAIR, single channel). Inputs must be FLAIR named `*_0000.*`.
- Built on CUDA 11.8; very recent GPUs may need a newer CUDA/driver. If `torch.cuda.is_available()` is `False` on a GPU node, try an older card.

## Troubleshooting

- "0 cases found" / dataset not found: ensure inputs are named `*_0000.*` and `--input_dir` points at the folder containing them.
- "No such file or directory" inside Apptainer: the path isn't mounted; add `--bind /that/path:/that/path`.
- CUDA out of memory: use a smaller input or a GPU with more memory.