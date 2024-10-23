import os
import time
import torch
import monai
import numpy as np
from pickle import load
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join as pjoin
from typing import Optional, Union, List, Dict, Generator
from monai.config.type_definitions import NdarrayOrTensor

from conflunet.dataloading.dataloaders import get_test_dataloader
from conflunet.training.utils import get_default_device
from conflunet.postprocessing.basic_postprocessor import Postprocessor
from conflunet.utilities.planning_and_configuration import PlansManagerInstanceSeg


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
            model: torch.nn.Module,
            postprocessor: Optional[Postprocessor] = None,
            output_dir: Optional[str] = None,
            preprocessed_files_dir: Optional[str] = None,
            num_workers: int = 0,
            save_only_instance_segmentation: bool = True,
            convert_to_original_shape: bool = False,
            verbose: bool = True
    ):
        super(Predictor, self).__init__()
        self.plans_manager = plans_manager
        self.configuration = plans_manager.get_configuration('3d_fullres')
        self.n_channels = len(plans_manager.foreground_intensity_properties_per_channel)
        self.model = model
        self.device = get_default_device()
        self.postprocessor = postprocessor
        self.output_dir = output_dir
        if preprocessed_files_dir is not None:
            self.preprocessed_files_dir = preprocessed_files_dir
        elif self.output_dir is not None:
            self.preprocessed_files_dir = pjoin(self.output_dir, 'preprocessed_files')
        else:
            self.preprocessed_files_dir = '.tmp'
            os.makedirs(self.preprocessed_files_dir, exist_ok=True)
        self.num_workers = num_workers
        self.configuration = plans_manager.get_configuration('3d_fullres')
        self.patch_size = self.configuration.patch_size
        self.batch_size = self.configuration.batch_size
        self.act = torch.nn.Softmax(dim=1)
        self.convert_to_original_shape = False
        self.save_only_instance_segmentation = save_only_instance_segmentation
        self.verbose = verbose

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
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if affine is None:
            affine = np.eye(4)

        file_name = pjoin(self.output_dir, f"{name}.nii.gz")
        self.vprint(f"[INFO] Saving {file_name}")
        nib.save(nib.Nifti1Image(data, affine), file_name)

    def save_predictions(self, output: Dict[str, NdarrayOrTensor]) -> None:
        actual_output = {k: v for k, v in output.items() if k in BASE_OUTPUT and v is not None}
        name = output['name']
        properties = output['properties']

        for file_name, pred in actual_output.items():
            if self.save_only_instance_segmentation and file_name != 'instance_seg_pred':
                continue

            affine = np.eye(4)
            if self.convert_to_original_shape:
                # TODO: retrieve the original affine from the sitk properties stuff
                raise NotImplementedError

            file_name = f"{name}_{file_name}_{self.postprocessor.name}"
            self.save_file(pred, file_name, affine)

    def preprocess_files(self, raw_files: Union[List, str]) -> None:
        # TODO: implement this method
        # set self.preprocessed_files_dir
        raise NotImplementedError

    def convert_to_original_size(self, output: Dict[str, NdarrayOrTensor]) -> Dict[str, NdarrayOrTensor]:
        if not self.convert_to_original_shape:
            return output
        # TODO: implement this method
        raise NotImplementedError

    def predict_from_raw_files(self, raw_files: Union[List, str], model: torch.nn.Module = None) -> None:
        if model is not None:
            self.model = model

        self.preprocess_files(raw_files)
        self.convert_to_original_shape = True
        self.predict_from_preprocessed_dir(self.model)
        os.removedirs(self.preprocessed_files_dir)

    def predict_from_raw_input_dir(self, input_dir: str, model: torch.nn.Module = None) -> None:
        if model is not None:
            self.model = model

        self.predict_from_raw_files(os.listdir(input_dir), self.model)

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
        for output in self.get_predictions_loader(dataloader, model):
            self.save_predictions(output)

    def get_predictions_loader(self, dataloader: monai.data.DataLoader,
                              model: torch.nn.Module = None,
                              max_workers: int = 4) -> Generator[Dict, None, None]:
        if model is not None:
            self.model = model

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
                prediction_result = self.predict_batch(batch, self.model)
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

    def predict_batch(self, batch: dict, model: torch.nn.Module = None) -> Dict[str, NdarrayOrTensor]:
        raise NotImplementedError

