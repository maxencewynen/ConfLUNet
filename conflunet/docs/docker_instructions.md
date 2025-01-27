# Using ConfLUNet with Docker

This guide explains how to use ConfLUNet through Docker containers. Docker provides a consistent environment for running ConfLUNet, ensuring all dependencies are properly configured regardless of your host system.

## System Requirements

The Docker container requires:
- Docker installed on your system
- NVIDIA Docker runtime for GPU support
- CUDA-capable GPU with appropriate drivers installed
- Sufficient disk space for the Docker image and data
- Adequate shared memory (the recommended 40GB is an approximate value and may need adjustment based on your specific dataset size and processing requirements)

## Building the Docker Image

You can either pull the pre-built image from Docker Hub or build it yourself from the source.

### Using Pre-built Image

```bash
docker pull petermcgor/conflunet:0.2.0
```

### Building from Source

From the root directory of the ConfLUNet repository:
```bash
docker build -t conflunet .
```

## Running ConfLUNet with Docker

The container requires mounting a volume for data persistence and GPU access. The basic command structure is:

```bash
docker run -it --gpus all --shm-size=40gb \
  -v /path/to/your/data:/opt/conflunet_resources \
  petermcgor/conflunet:0.2.0 [command]
```

Where:
- `--gpus all`: Enables GPU access
- `--shm-size=40gb`: Sets shared memory size. This value is approximate and may need adjustment depending on your specific use case, dataset size, and processing requirements
- `-v /path/to/your/data:/opt/conflunet_resources`: Mounts your local data directory. Important: Your mounted volume MUST contain the following directory structure as required by nnUNet:
  ```
  /path/to/your/data/
  ├── nnUNet_raw/         # Raw data following nnUNet format
  ├── nnUNet_preprocessed/ # Will contain preprocessed data
  └── nnUNet_results/     # Will store training results
  ```

The mounted volume must strictly follow this structure, as ConfLUNet relies on nnUNet's directory organization for proper functioning. For detailed information about the required directory structure and file organization, please refer to the nnUNet documentation.

### Example Commands

Running preprocessing:
```bash
docker run -it --gpus all --shm-size=40gb \
  -v /path/to/your/data:/opt/conflunet_resources \
  petermcgor/conflunet:0.2.0 \
  conflunet_plan_and_preprocess --dataset_id [ID] --check_dataset_integrity
```

Training a model:
```bash
docker run -it --gpus all --shm-size=40gb \
  -v /path/to/your/data:/opt/conflunet_resources \
  petermcgor/conflunet:0.2.0 \
  conflunet_train --dataset_id [ID] --fold [FOLD] --model_name [NAME]
```

To disable Weights & Biases logging:
```bash
docker run -it --gpus all --shm-size=40gb \
  -v /path/to/your/data:/opt/conflunet_resources \
  petermcgor/conflunet:0.2.0 \
  conflunet_train --dataset_id [ID] --fold [FOLD] --model_name [NAME] --wandb_ignore
```

### Interactive Shell

For interactive use:
```bash
docker run -it --gpus all --shm-size=40gb \
  -v /path/to/your/data:/opt/conflunet_resources \
  petermcgor/conflunet:0.2.0 bash
```

## Data Organization

The container expects your data to follow the nnUNet dataset format. Mount your data directory to `/opt/conflunet_resources` in the container, ensuring it contains the required nnUNet directory structure:
- `nnUNet_raw`: Raw data directory
- `nnUNet_preprocessed`: Preprocessed data directory
- `nnUNet_results`: Results directory

[Placeholder: Additional details about specific dataset organization requirements]

## Troubleshooting

[Placeholder: Common issues and their solutions]

## Performance Considerations

[Placeholder: Guidelines for optimal performance, memory management, etc.]