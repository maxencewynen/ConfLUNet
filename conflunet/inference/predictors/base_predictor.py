import os
import time
import torch
import monai
import numpy as np
from pickle import load
import nibabel as nib
from os.path import join as pjoin
from typing import Optional, Union, List, Dict, Generator
from monai.config.type_definitions import NdarrayOrTensor

from conflunet.dataloading.dataloaders import get_test_dataloader
from conflunet.training.utils import get_default_device
from conflunet.postprocessing.basic_postprocessor import BasicPostprocessor
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


class BasePredictor:
    def __init__(
            self,
            plans_manager: PlansManagerInstanceSeg,
            model: torch.nn.Module,
            postprocessor: Optional[BasicPostprocessor] = None,
            output_dir: Optional[str] = None,
            preprocessed_files_dir: Optional[str] = None,
            num_workers: int = 0,
            save_only_instance_segmentation: bool = True
    ):
        super(BasePredictor, self).__init__()
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
        self.patch_size = self.configuration.patch_size#(96,96,96)#
        self.batch_size = self.configuration.batch_size#4#
        self.act = torch.nn.Softmax(dim=1)
        self.should_revert_preprocessing = False
        self.save_only_instance_segmentation = save_only_instance_segmentation

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
        print(f"[INFO] Saving {file_name}")
        nib.save(nib.Nifti1Image(data, affine), file_name)

    def save_predictions(self, output: Dict[str, NdarrayOrTensor]) -> None:
        actual_output = {k: v for k, v in output.items() if k in BASE_OUTPUT and v is not None}
        name = output['name']
        properties = output['properties']

        for file_name, pred in actual_output.items():
            if self.save_only_instance_segmentation and file_name != 'instance_seg_pred':
                continue

            affine = np.eye(4)
            if self.should_revert_preprocessing:
                # TODO: retrieve the original affine from the sitk properties stuff
                raise NotImplementedError

            file_name = f"{name}_{file_name}_{self.postprocessor.name}"
            self.save_file(pred, file_name, affine)

    def preprocess_files(self, raw_files: Union[List, str]) -> None:
        # TODO: implement this method
        # set self.preprocessed_files_dir
        raise NotImplementedError

    def convert_to_original_size(self, output: Dict[str, NdarrayOrTensor]) -> Dict[str, NdarrayOrTensor]:
        if not self.should_revert_preprocessing:
            return output
        # TODO: implement this method
        raise NotImplementedError

    def predict_from_raw_files(self, raw_files: Union[List, str], model: torch.nn.Module = None) -> None:
        if model is not None:
            self.model = model

        self.preprocess_files(raw_files)
        self.should_revert_preprocessing = True
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
                              model: torch.nn.Module = None) -> Generator[Dict, None, None]:
        if model is not None:
            self.model = model

        for i, batch in enumerate(dataloader):
            print(f"\n\n>>>>> Starting prediction of case {batch['name'][0]} <<<<<")
            with open(batch['properties_file'][0], 'rb') as f:
                properties = load(f)
            start = time.time()
            ret =  self.convert_to_original_size(
                self.postprocessor({
                    **self.predict_batch(batch, self.model),
                    'name': batch['name'][0],
                    'properties': properties
                    }
                )
            )
            elapsed = time.time() - start
            print(f"[INFO] Full prediction of {batch['name'][0]} took {elapsed:.2f} seconds")
            yield ret

    def predict_batch(self, batch: dict, model: torch.nn.Module = None) -> Dict[str, NdarrayOrTensor]:
        raise NotImplementedError

