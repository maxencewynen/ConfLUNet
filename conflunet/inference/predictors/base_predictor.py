import os
import time
import torch
import monai
import shutil
import warnings
import SimpleITK as sitk
import numpy as np
from pickle import load
import nibabel as nib
from os.path import join as pjoin
from os.path import exists as pexists
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union, List, Dict, Generator
from monai.config.type_definitions import NdarrayOrTensor
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

from conflunet.dataloading.dataloaders import get_test_dataloader
from conflunet.training.utils import get_default_device
from conflunet.postprocessing.basic_postprocessor import Postprocessor
from conflunet.utilities.planning_and_configuration import PlansManagerInstanceSeg
from conflunet.architecture.utils import load_model
from conflunet.preprocessing.preprocess import preprocess_dataset


BASE_OUTPUT = {
    'instance_seg_pred': None,
    'semantic_pred_proba': None,
    'semantic_pred_binary': None,
    'center_pred': None,
    'center_pred_binary': None,
    'offsets_pred': None,
    'offsets_pred_x': None,
    'offsets_pred_y': None,
    'offsets_pred_z': None,
}


class Predictor:
    def __init__(
            self,
            plans_manager: PlansManagerInstanceSeg,
            model: Union[Union[torch.nn.Module, str], List[Union[str, torch.nn.Module]]],
            postprocessor: Optional[Postprocessor] = None,
            output_dir: Optional[str] = None,
            preprocessed_files_dir: Optional[str] = None,
            num_workers: int = 0,
            save_only_instance_segmentation: bool = True,
            convert_to_original_shape: bool = False,
            semantic: bool = False,
            verbose: bool = True
    ):
        super(Predictor, self).__init__()
        self.plans_manager = plans_manager
        self.configuration = plans_manager.get_configuration('3d_fullres')
        self.n_channels = len(plans_manager.foreground_intensity_properties_per_channel)
        self.device = get_default_device()
        self.postprocessor = postprocessor
        self.output_dir = output_dir
        self.semantic = semantic
        if preprocessed_files_dir is not None:
            self.preprocessed_files_dir = preprocessed_files_dir
        elif self.output_dir is not None:
            self.preprocessed_files_dir = pjoin(self.output_dir, 'preprocessed')
        else:
            self.preprocessed_files_dir = '.tmp'
        os.makedirs(self.preprocessed_files_dir, exist_ok=True)
        self.num_workers = num_workers
        self.configuration = plans_manager.get_configuration('3d_fullres')
        self.patch_size = self.configuration.patch_size
        self.batch_size = self.configuration.batch_size
        self.act = torch.nn.Softmax(dim=1)
        self.convert_to_original_shape = convert_to_original_shape
        self.save_only_instance_segmentation = save_only_instance_segmentation
        self.verbose = verbose
        self.model = self._set_model(model)
        if isinstance(self.model, torch.nn.Module):
            self.model = {None: self.model}

    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def save_file(self, data: NdarrayOrTensor, name: str, affine: Optional[NdarrayOrTensor] = None) -> None:
        if self.output_dir is None:
            raise ValueError("Output directory must be provided. Set 'output_dir' attribute.")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        if data is None:
            return
        if affine is None:
            affine = np.eye(4)

        file_name = pjoin(self.output_dir, f"{name}.nii.gz")
        self.vprint(f"[INFO] Saving {file_name}")
        nib.save(nib.Nifti1Image(data, affine), file_name)

    @staticmethod
    def sitk_save(data: np.ndarray, output_fname: str, properties: Dict) -> None:
        assert data.ndim == 3, ('data must be 3d. If you are exporting a 2d segmentation, please provide it as '
                               'shape 1,x,y')
        output_dimension = len(properties['sitk_stuff']['spacing'])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            data = data[0]

        itk_image = sitk.GetImageFromArray(data)
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])

        sitk.WriteImage(itk_image, output_fname, True)

    def save_predictions(self, output: Dict[str, NdarrayOrTensor]) -> None:
        actual_output = {k: v for k, v in output.items() if k in BASE_OUTPUT and v is not None}
        name = output['name']
        properties = output['properties']

        for file_name, pred in actual_output.items():
            if file_name == 'offsets_pred': continue
            if self.save_only_instance_segmentation and file_name != 'instance_seg_pred':
                continue

            pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred

            file_name = f"{name}_{file_name}_{self.postprocessor.name}"
            if self.convert_to_original_shape:
                self.sitk_save(pred, pjoin(self.output_dir, f"{file_name}.nii.gz"), properties)

            else:
                affine = np.eye(4)
                self.save_file(pred, file_name, affine)

    def preprocess(self, input_dir: str) -> None:
        preprocess_dataset(
            dataset_id=self.plans_manager.dataset_id,
            configurations=['3d_fullres'],
            num_processes=[self.num_workers],
            inference=True,
            plans_manager=self.plans_manager,
            output_dir_for_inference=self.preprocessed_files_dir,
            input_dir=input_dir,
            verbose=self.verbose
        )

    def convert_to_original_size(self, output: Dict[str, NdarrayOrTensor]) -> Dict[str, NdarrayOrTensor]:
        if not self.convert_to_original_shape:
            return output
        self.vprint(f"[INFO] Converting to original size for {output['name']}")
        properties = output['properties']
        spacing_transposed = [properties['spacing'][i] for i in self.plans_manager.transpose_forward]
        current_spacing = self.configuration.spacing if \
            len(self.configuration.spacing) == \
            len(properties['shape_after_cropping_and_before_resampling']) else \
            [spacing_transposed[0], *self.configuration.spacing]

        for key, value in output.items():
            if value is None:
                continue
            if key in ("semantic_pred_binary", "center_pred_binary", "instance_seg_pred"):
                resample_fn = self.configuration.resampling_fn_seg
            else:
                resample_fn = self.configuration.resampling_fn_probabilities

            if key != 'offsets_pred' and (isinstance(value, torch.Tensor) or isinstance(value, np.ndarray)):
                # Here we undo the preprocessing done by nnUNet (cf. conflunet/preprocessing/preprocessors.py)
                # As the preprocessing went transpose -> crop -> resample, we need to revert this process by doing
                # resample -> crop -> transpose
                this_output = value.to('cpu') if isinstance(value, torch.Tensor) else value
                # First make sure that the output is in the correct shape for the nnUNet resampling function
                should_i_squeeze = False
                should_i_unsqueeze = False
                if len(this_output.shape) == 3:
                    this_output = torch.unsqueeze(this_output, 0)
                    should_i_squeeze = True
                elif len(this_output.shape) == 5:
                    this_output = torch.squeeze(this_output, 0)
                    should_i_unsqueeze = True
                elif len(this_output.shape) != 4:
                    raise ValueError(f"Output shape must be 3, 4 or 5 but got {len(this_output.shape)}")

                # Now we resample the output to the shape before cropping
                this_output = resample_fn(this_output,
                                          properties['shape_after_cropping_and_before_resampling'],
                                          current_spacing,
                                          [properties['spacing'][i] for i in self.plans_manager.transpose_forward])

                # Now that is correctly resampled, we revert the cropping by padding zeros to the original shape
                slicer = bounding_box_to_slice(properties['bbox_used_for_cropping'])
                this_output = np.squeeze(this_output, 0)
                this_output_reverted_cropping = np.zeros(properties['shape_before_cropping'])
                this_output_reverted_cropping[slicer] = this_output
                del this_output

                # Finally, we revert the transpose operation
                this_output_reverted_cropping = this_output_reverted_cropping.transpose(self.plans_manager.transpose_backward)
                # maybe convert to torch tensor
                if isinstance(output[key], torch.Tensor) and not isinstance(this_output_reverted_cropping, torch.Tensor):
                    # As the output is likely to be way larger than the input, we put it back on the cpu to avoid
                    # any GPU memory issues
                    this_output_reverted_cropping = torch.from_numpy(this_output_reverted_cropping).to('cpu')

                # If we squeezed/unsqueezed the tensor at the beginning, we need to unsqueeze/squeeze it back to the
                # original shape
                if should_i_squeeze:
                    this_output_reverted_cropping = torch.squeeze(this_output_reverted_cropping, 0)
                elif should_i_unsqueeze:
                    this_output_reverted_cropping = torch.unsqueeze(this_output_reverted_cropping, 0)

                output[key] = this_output_reverted_cropping

            elif key == 'offsets_pred':
                # Unfortunately, only this transpose_backward is supported because of the conversion to original shape of
                # the predicted offsets. This can be changed in the future, because for the moment it is not a priority.
                if not self.semantic and tuple(self.plans_manager.transpose_backward) != (1, 2, 0):
                    warnings.warn(f"Only transpose_backward = (1, 2, 0) is supported, got "
                                  f"{tuple(self.plans_manager.transpose_backward)}")
                if not self.semantic and tuple(self.plans_manager.transpose_forward) != (2, 0, 1):
                    warnings.warn(f"Only transpose_forward = (2, 0, 1) is supported, got "
                                  f"{tuple(self.plans_manager.transpose_forward)}")
                # This is a special case because the offsets values actually correspond to different axes.
                value = torch.squeeze(value, 0)
                slicer = bounding_box_to_slice(properties['bbox_used_for_cropping'])
                value = value.to('cpu') if isinstance(value, torch.Tensor) else value

                converted_output = []
                # We need to individually undo the preprocessing for each axis of the offsets (x, y, z), and stack
                # them in the correct order so that it makes sense in the final prediction.
                # Here set the correct order to be (0, 2, 1) because we assume a specific transpose_backward tuple
                # (2, 1, 0) (as stated above). How to generalize this is still an open question for
                # which I don't have the time to think about for the moment.
                # To be honest, I don't fully understand the logic behind this, so I can't provide a better solution
                # for the moment. I just know it works like this :-).
                for dim in (0, 2, 1):
                    # First make sure that the output is in the correct shape for the nnUNet resampling function
                    this_output_dim = torch.unsqueeze(value[dim], 0)

                    # Now we resample the output to the shape before cropping
                    this_output_dim = resample_fn(this_output_dim,
                                                  properties['shape_after_cropping_and_before_resampling'],
                                                  current_spacing,
                                                  [properties['spacing'][i] for i in
                                                   self.plans_manager.transpose_forward])

                    # Now that is correctly resampled, we revert the cropping by padding zeros to the original shape
                    this_output_dim = np.squeeze(this_output_dim, 0)
                    this_output_dim_reverted_cropping = np.zeros(properties['shape_before_cropping'])
                    this_output_dim_reverted_cropping[slicer] = this_output_dim

                    # Finally, we revert the transpose operation
                    this_output_dim_reverted_cropping = this_output_dim_reverted_cropping.transpose(self.plans_manager.transpose_backward)
                    converted_output.append(this_output_dim_reverted_cropping)
                    del this_output_dim, this_output_dim_reverted_cropping

                # Stack the converted outputs in the correct order
                converted_output = np.stack(converted_output)
                if isinstance(output[key], torch.Tensor) and not isinstance(converted_output, torch.Tensor):
                    converted_output = torch.from_numpy(converted_output).to('cpu')

                converted_output = torch.unsqueeze(converted_output, 0)
                output[key] = converted_output

        return output

    def predict_from_raw_files(self, raw_files: Union[List, str], model: torch.nn.Module = None) -> None:
        raise NotImplementedError

    def predict_from_raw_input_dir(self, input_dir: str, model: torch.nn.Module = None) -> None:
        if model is not None:
            self.model = model

        self.preprocess(input_dir)
        self.convert_to_original_shape = True
        self.predict_from_preprocessed_dir(self.model)
        shutil.rmtree(self.preprocessed_files_dir)

    def get_dataloader(self) -> monai.data.DataLoader:
        return get_test_dataloader(folder=self.preprocessed_files_dir, num_workers=self.num_workers, batch_size=1)

    def predict_from_preprocessed_dir(self, model: torch.nn.Module = None) -> None:
        # Saves the predictions to the output directory
        if model is not None:
            self.model = model

        dataloader = self.get_dataloader()
        self.predict_from_dataloader(dataloader, self.model)

    def predict_from_dataloader(self, dataloader: monai.data.DataLoader,
                                model: torch.nn.Module = None) -> None:
        prediction_loader = self.get_predictions_loader(dataloader, model)
        for output in prediction_loader:
            print("\n\n>>>>> Saving predictions <<<<<")
            self.save_predictions(output)

    def get_predictions_loader(self, dataloader: monai.data.DataLoader,
                              model: Union[torch.nn.Module, Dict[int, torch.nn.Module]] = None,
                              max_workers: int = 4) -> Generator[Dict, None, None]:
        if model is not None:
            self.model = model
        if isinstance(self.model, torch.nn.Module):
            self.model = {None: self.model}

        futures = []
        next_result_idx = 0  # Tracks the next result to yield in the correct order
        results_dict = {}    # Dictionary to hold completed results, keyed by index

        # Use a thread pool executor for postprocessing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, batch in enumerate(dataloader):
                print(f"\n\n>>>>> Starting prediction of case {batch['name'][0]} <<<<<")
                with open(batch['properties_file'][0], 'rb') as f:
                    properties = load(f)

                # Perform the prediction sequentially
                start_prediction = time.time()
                if isinstance(self.model, dict):
                    # Ensemble prediction
                    prediction_result = BASE_OUTPUT.copy()
                    for j, model in self.model.items():
                        this_model_result = self.predict_batch(batch, model, model_index=j)
                        for key, value in this_model_result.items():
                            if value is None:
                                continue
                            if prediction_result[key] is None:
                                prediction_result[key] = value
                            else:
                                prediction_result[key] += value
                    for key, value in prediction_result.items():
                        if value is not None:
                            prediction_result[key] = value / len(self.model)
                    del this_model_result
                elif isinstance(self.model, torch.nn.Module):
                    # Single model prediction
                    prediction_result = self.predict_batch(batch, self.model)
                else:
                    raise ValueError(f"Model must be a torch.nn.Module or a dictionary of (int, torch.nn.Module)"
                                     f"but got {self.model}")
                print(f"[INFO] Prediction for {batch['name'][0]} took {time.time() - start_prediction:.2f} seconds")

                # Submit postprocessing tasks to be executed in parallel
                future = executor.submit(self.process_result, prediction_result, batch['name'][0], properties)
                futures.append((i, future))  # Store the index along with the future

            # Process the futures as they complete
            for future in as_completed(future for _, future in futures):
                # Find the index of the completed task
                idx = next(idx for idx, f in futures if f == future)

                # Store the result in the dictionary
                results_dict[idx] = future.result()

                # Yield any ready results in order
                while next_result_idx in results_dict:
                    yield results_dict.pop(next_result_idx)
                    next_result_idx += 1

    def process_result(self, prediction_result, name, properties):
        print(f"[INFO] START Postprocessing for {name}")
        # Postprocessing task to be run in parallel
        start_postprocessing = time.time()
        ret = self.convert_to_original_size(
                self.postprocessor({
                    **prediction_result,
                    'name': name,
                    'properties': properties
                })
        )
        elapsed_postprocessing = time.time() - start_postprocessing
        print(f"[INFO] STOP Postprocessing for {name} took {elapsed_postprocessing:.2f} seconds")
        return ret

    def predict_batch(self, batch: dict, model: torch.nn.Module = None, model_index=None) -> Dict[str, NdarrayOrTensor]:
        raise NotImplementedError

    def _set_model(self, model: Union[Union[torch.nn.Module, str], List[Union[str, torch.nn.Module]]]
                   ) -> Union[torch.nn.Module, Dict[int, torch.nn.Module]]:
        if model is None:
            return {}
        elif isinstance(model, str):
            assert pexists(model), f"Model file {model} does not exist"
            return load_model(self.configuration, model, self.n_channels, semantic=self.semantic)
        elif isinstance(model, torch.nn.Module):
            return model
        elif isinstance(model, list):
            return {i: self._set_model(m) for i, m in enumerate(model)}
        else:
            raise ValueError(f"Model must be a str, torch.nn.Module or a list of str or torch.nn.Module but got {model}")

