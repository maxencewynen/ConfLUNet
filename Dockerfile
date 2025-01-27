# Start with PyTorch base image with CUDA support for GPU acceleration
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Avoid timezone prompt during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


# Copy the project files
RUN mkdir /conflunet
WORKDIR /conflunet
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Set up environment variables required by ConfLUNet/nnUNet
ARG resources="/opt/conflunet_resources"
ENV nnUNet_raw=$resources"/nnUNet_raw" nnUNet_preprocessed=$resources"/nnUNet_preprocessed" nnUNet_results=$resources"/nnUNet_results"

