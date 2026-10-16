# ConfLUNet: Multiple sclerosis lesion instance segmentation

## What is ConfLUNet?

ConfLUNet is a deep learning model that performs instance segmentation of multiple sclerosis lesions in FLAIR MRI images. The model is based on the [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/master) framework. The model is trained on a proprietary dataset.

## How to get started?

* [Installation instruction](conflunet/docs/installation_instructions.md) or follow the docker instructions below to avoid installing the dependencies yourself.
* [Docker instructions](conflunet/docs/docker_instructions.md) — the Docker image ships with a **default pre-trained model** so you can run inference out of the box (no training or data setup required).
* [Dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) (following nnUNet)  
* [Usage instructions](conflunet/docs/how_to_use_conflunet.md) 


## Contributing

Feel free to open an issue or pull request if you have any suggestions, questions, or improvements.


## Notes

* **Dataset id `321` is reserved for the bundled model.** Inside the Docker image, running inference with `--dataset_id 321` always loads the default pre-trained model baked into the image (`/opt/conflunet_default_inference_model`); a dataset of your own numbered `321` will be overridden for inference. Use a different dataset id for your own data. Note that `--dataset_id 321` currently assumes the bundled model is present, so it is intended for use inside the Docker/Singularity image. See the [Docker instructions](conflunet/docs/docker_instructions.md) for details.

## License

This project is licensed under the MIT LICENSE - see the LICENSE file for details.

## Acknowledgements
<img src="conflunet/docs/assets/uclouvain_logo.png" height="100px" />
<img src="conflunet/docs/assets/logo_UNIL.png" height="100px" />
<img src="conflunet/docs/assets/trail-logo.jpg" height="100px" />
