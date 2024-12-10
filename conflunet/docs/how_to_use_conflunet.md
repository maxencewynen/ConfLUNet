# How to run ConfLUNet on a new dataset

ConfLUNet's chore is heavily based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet)'s framework (if you don't already know about it, I would definitely recommend you take a look at what's done over there!). Although ConfLUNet was specifically created for instance segmentation of lesions in brain MRI images of multiple sclerosis patients, the fact it relies on nnUNet makes it easily expandable to other use cases in the future.

As with nnUNet, given some dataset, ConfLUNet fully automatically configures an entire segmentation pipeline that matches its properties. ConfLUNet covers the entire pipeline, from preprocessing to model training, postprocessing all the way to ensembling. After running ConfLUNet, the trained model(s) can be applied to the test cases for inference.

### Dataset Format
ConfLUNet expects the data to be structured the same way as for nnUNet. Please go and read [nnUNet's dedicated guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) to how to structure and organize your dataset properly.

### Experiment planning and preprocessing
ConfLUNet uses bricks of the nnUNet framework for this part. nnU-Net will extract a dataset fingerprint (a set of dataset-specific properties such as image sizes, voxel spacings, intensity information etc). Though this information is used to design three U-Net configurations, ConfLUNet only uses one configuration (`3d_fullres`). During pre-processing, ConfLUNet adds some specific information to the data (mostly information related to instances, small lesions and confluent lesions).

To run fingerprint extraction, experiment planning and preprocessing you will have to use:
```commandline
conflunet_plan_and_preprocess --dataset_id DATASET_ID --check_dataset_integrity --num_processes NUM_PROCESSES
```
Same as for nnUNet, we recommend using the --check_dataset_integrity whenever it's the first time you run this command. This will check for some of the most common error sources!

### Model training
#### Overview
Similar to nnUNet, ConfLUNet is made to be trained using a 5-fold cross-validation strategy over the training cases. This is a natural way of obtaining a good model ensemble (average the output of these 5 models for prediction) to boost performance.

You can influence the splits ConfLUNet uses for 5-fold cross-validation (see [nnUNet's documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/manual_data_splits.md) about this). Currently, training a single model on all training cases is not supported.

Training models is done with the conflunet_train command. The general structure of the command is:
```commandline
conflunet_train --dataset_id DATASET_ID --fold FOLD --model_name MODEL_NAME [additional options, see -h]
```
`MODEL_NAME` is the name you give to the model.
ConfLUNet uses wandb ([weights and biases](https://wandb.ai/)) as a logger by default. If you do not wish to use this option, you can disable it with the `--wandb_ignore` flag.

For comparison with the same model architecture but a single semantic output, you may want to use the `--semantic` flag.

See `conflunet_train -h` for additional options.

#### Checkpoint saving
Checkpoints are saved in the `nnUNet_results/Dataset[DATASET_ID]_[DATASET_NAME]/[MODEL_NAME]` folder. Two kinds of checkpoints are saved for each fold: `checkpoint_final.pth` (last checkpoint) and `checkpoint_best_[METRIC]_[POSTPROCESSOR_NAME].pth`.
- For ConfLUNet, METRIC is always Panoptic_Quality, and the POSTPROCESSOR is always `ConfLUNet`.
- For models trained with the `--semantic` flag, METRIC can either be `Dice_score` or `Normalized_Dice`. The postprocessor is either `ACLS` or `CC` (see below for details on post-processing).


### Post-processing
Post-processing is done automatically on running the inference as well as during the training to be able to compute instance segmentation metrics and choose the best checkpoints based on this value.
Both ConfLUNet and its semantic counterpart needs post-processing to transform the model's output into instance segmentation. Three different postprocessors are currently implemented (refer to `conflunet.postprocessing` for more details):
1. ConfLUNet's post-processor (name: `ConfLUNet`). Requires the model output to be constituted of a semantic output, a center heatmap and an offsets map. 
2. Post-processors based on semantic models:
   - `ACLS` (Automated Confluent Lesion Splitting) is based on Dworkin et al.'s work
   - `CC` (Connected components)

### Inference
Remember that the data located in the input folder must have the file endings as the dataset you trained the model on and must adhere to the nnU-Net naming scheme for image files (see nnUNet's [dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) and [inference data format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format_inference.md)!)

To run inference on raw data (ideally skull-stripped!):
```commandline
conflunet_predict --dataset_id DATASET_ID --input_dir INPUT_DIR --output_dir OUTPUT_DIR --model MODEL [OPTIONAL --semantic --postprocessor POSTPROCESSOR]
```
Where `DATASET_ID` is the id of the dataset used at training time and `MODEL` is the name of the model you gave it during training.
This will perform several steps: 
1. Apply preprocessing following nnUNet's `plans.json` file
2. Run the model on patches of the images using a sliding window strategy
3. Apply the postprocessor's postprocessing steps (see above for more information on post-processors)
4. Save the results in `OUTPUT_DIR`

## How to run inference with pretrained models
[Dedicated documentation]()

## How to Deploy and Run Inference with YOUR Pretrained Models
TODO !