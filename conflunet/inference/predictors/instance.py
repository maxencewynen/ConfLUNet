import torch
import time
from typing import Optional, Dict, Union

from monai.inferers import sliding_window_inference
from monai.config.type_definitions import NdarrayOrTensor

from conflunet.inference.predictors.base_predictor import Predictor, BASE_OUTPUT
from conflunet.postprocessing.basic_postprocessor import Postprocessor
from conflunet.postprocessing.instance import ConfLUNetPostprocessor
from conflunet.utilities.planning_and_configuration import PlansManagerInstanceSeg


class ConfLUNetPredictor(Predictor):
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
        super(ConfLUNetPredictor, self).__init__(
            plans_manager=plans_manager,
            model=model,
            semantic=False,
            postprocessor=postprocessor,
            output_dir=output_dir,
            preprocessed_files_dir=preprocessed_files_dir,
            num_workers=num_workers,
            save_only_instance_segmentation=save_only_instance_segmentation,
            convert_to_original_shape=convert_to_original_shape,
            verbose=verbose
        )

    def predict_batch(self, batch: dict, model: Union[Dict[int, torch.nn.Module], torch.nn.Module] = None,
                      model_index: int = None) -> Dict[str, NdarrayOrTensor]:
        if model is not None:
            self.model[model_index] = model
        assert self.model[model_index] is not None, "Model must be provided"
        device = self.device

        self.model[model_index].to(device)
        self.model[model_index].eval()

        img = batch["img"].to(device)
        assert img.shape[0] == 1, "Batch size for original batch must be 1"

        spatial_shape = img.shape[-3:]
        brainmask = batch["brainmask"].to(device) if "brainmask" in batch.keys() else torch.ones(spatial_shape).to(
            device)

        patch_size = self.patch_size
        start = time.time()
        with torch.no_grad():
            outputs = sliding_window_inference(img, patch_size, self.batch_size,
                                               self.model[model_index], mode='gaussian', overlap=0.5)
        self.vprint(f"[INFO] Sliding window inference took {time.time() - start:.2f} seconds")
        if not isinstance(outputs, tuple) and len(outputs) != 3:
            raise ValueError("The model should output a tuple of 3 tensors")

        semantic_pred_proba, heatmap_pred, offsets_pred = outputs
        heatmap_pred = heatmap_pred.half()
        offsets_pred = offsets_pred.half()

        semantic_pred_proba = torch.squeeze(self.act(semantic_pred_proba)[0, 1]) * brainmask

        del img, outputs
        torch.cuda.empty_cache()

        output_dict = BASE_OUTPUT.copy()
        output_dict['semantic_pred_proba'] = semantic_pred_proba
        output_dict['center_pred'] = heatmap_pred
        output_dict['offsets_pred'] = offsets_pred

        assert 'semantic_pred_binary' in output_dict.keys(), "Postprocessor must return 'semantic_pred_binary'"
        assert 'instance_seg_pred' in output_dict.keys(), "Postprocessor must return 'instance_seg_pred'"
        return output_dict


if __name__ == '__main__':
    from conflunet.utilities.planning_and_configuration import load_dataset_and_configuration
    from conflunet.architecture.nnconflunet import nnConfLUNet
    from pickle import load
    from torch import nn
    import torch
    import numpy as np
    from conflunet.architecture.utils import load_model
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(320)

    # model = ConfLUNet(1,2, scale_offsets=20)
    # path_to_model = "/home/mwynen/Downloads/best_DSC_SO_A_L1_e-5_h1200o0.3_S20_seed1.pth"
    # model.load_state_dict(torch.load(path_to_model))
    # model = UNet3D(1,2)
    # path_to_model = "/home/mwynen/Downloads/best_DSC_SS_lre-5_seed1.pth"
    # model.load_state_dict(torch.load(path_to_model))
    # model = DummySemanticProbabilityModel()
    # models = f"/home/mwynen/Dataset321_WMLIS/checkpoints/fold_0/checkpoint_best_Panoptic_Quality_ConfLUNet.pth"
    # models = load_model(configuration, models, n_channels, False)
    models = [f"/home/mwynen/Dataset321_WMLIS/checkpoints/fold_{f}/checkpoint_best_Panoptic_Quality_ConfLUNet.pth" for f in range(5)]


    p = ConfLUNetPredictor(
        plans_manager=plans_manager,
        model=models,
        postprocessor=ConfLUNetPostprocessor(
            minimum_instance_size=3,
            semantic_threshold=0.5,
            verbose=True
        ),
        output_dir='/home/mwynen/data/nnUNet/tmp/output_dir_test',
        # preprocessed_files_dir='/home/mwynen/data/nnUNet/tmp/preprocessed',
        num_workers=1,
        save_only_instance_segmentation=False
    )
    p.predict_from_raw_input_dir(input_dir='/home/mwynen/data/nnUNet/nnUNet_raw/Dataset321_WMLIS/imagesTs')
    # p.predict_from_preprocessed_dir()
    # dataloader = p.get_dataloader()
    # predictions_loader = p.get_predictions_loader(dataloader, p.model)
    # start = time.time()
    # for data_batch, predicted_batch in zip(dataloader, predictions_loader):
    #     print(data_batch['name'], predicted_batch['name'])
    # print(f"Total time (parallelized): {time.time() - start:.2f} seconds")
    # pass

    # def save_nii(data, properties, output_path):
    #     import nibabel as nib
    #     img = nib.Nifti1Image(data, np.eye(4))
    #     img.header.set_zooms(properties['spacing'])
    #     nib.save(img, output_path)
    # from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
    #
    # dataloader = p.get_dataloader()
    # for i, batch in enumerate(dataloader):
    #     if i == 0:
    #         continue
    #     with open(batch['properties_file'][0], 'rb') as f:
    #         properties = load(f)
    #     print(properties)
    #     spacing_transposed = [properties['spacing'][i] for i in plans_manager.transpose_forward]
    #     current_spacing = configuration.spacing if \
    #         len(configuration.spacing) == \
    #         len(properties['shape_after_cropping_and_before_resampling']) else \
    #         [spacing_transposed[0], *configuration.spacing]
    #     img = batch['img'][0].cpu().numpy()
    #     save_nii(np.squeeze(img), properties, '/home/mwynen/data/nnUNet/tmp/output_dir_test/TEST_0_original_img.nii.gz')
    #     resampled_img = configuration.resampling_fn_probabilities(img,
    #                                                               properties['shape_after_cropping_and_before_resampling'],
    #                                                               current_spacing,
    #                                                               [properties['spacing'][i] for i in plans_manager.transpose_forward])
    #     save_nii(np.squeeze(resampled_img), properties, '/home/mwynen/data/nnUNet/tmp/output_dir_test/TEST_1_resampled_img.nii.gz')
    #
    #     slicer = bounding_box_to_slice(properties['bbox_used_for_cropping'])
    #     slicer = tuple([slicer[i] for i in plans_manager.transpose_backward])
    #     resampled_img = np.squeeze(resampled_img, 0)
    #     resampled_img = resampled_img.transpose(plans_manager.transpose_backward)
    #     resampled_img_reverted_cropping = np.zeros(properties['shape_before_cropping'])
    #     resampled_img_reverted_cropping[slicer] = resampled_img
    #     p.sitk_save(resampled_img_reverted_cropping, '/home/mwynen/data/nnUNet/tmp/output_dir_test/TEST_2_resampled_img_reverted_cropping.nii.gz', properties)
    #
    #     break
