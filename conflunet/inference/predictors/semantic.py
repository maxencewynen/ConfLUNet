import torch
import time
from typing import Optional, Dict, Union

from monai.inferers import sliding_window_inference
from monai.config.type_definitions import NdarrayOrTensor

from conflunet.inference.predictors.base_predictor import Predictor, BASE_OUTPUT
from conflunet.postprocessing.basic_postprocessor import Postprocessor
from conflunet.postprocessing.semantic import ACLSPostprocessor
from conflunet.utilities.planning_and_configuration import PlansManagerInstanceSeg


class SemanticPredictor(Predictor):
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
        super(SemanticPredictor, self).__init__(
            plans_manager=plans_manager,
            model=model,
            semantic=True,
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
            outputs = sliding_window_inference(img, patch_size, self.batch_size, self.model[model_index],
                                               mode='gaussian', overlap=0.5)
        self.vprint(f"[INFO] Sliding window inference took {time.time() - start:.2f} seconds")

        if isinstance(outputs, tuple):
            raise ValueError("The model should output a single tensor, not a tuple of tensors")
        semantic_pred_proba = torch.squeeze(self.act(outputs)[0, 1]) * brainmask

        del img, outputs
        torch.cuda.empty_cache()

        output_dict = BASE_OUTPUT.copy()
        output_dict['semantic_pred_proba'] = semantic_pred_proba

        assert 'semantic_pred_binary' in output_dict.keys(), "Postprocessor must return 'semantic_pred_binary'"
        assert 'instance_seg_pred' in output_dict.keys(), "Postprocessor must return 'instance_seg_pred'"
        return output_dict


if __name__ == '__main__':
    from conflunet.utilities.planning_and_configuration import load_dataset_and_configuration
    from conflunet.architecture.conflunet import ConfLUNet, UNet3D
    from torch import nn
    import torch
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(321)

    # model = ConfLUNet(1,2, scale_offsets=20)
    # path_to_model = "/home/mwynen/Downloads/best_DSC_SO_A_L1_e-5_h1200o0.3_S20_seed1.pth"
    # model.load_state_dict(torch.load(path_to_model))
    # model = UNet3D(1,2)
    # path_to_model = "/home/mwynen/Downloads/best_DSC_SS_lre-5_seed1.pth"
    # model.load_state_dict(torch.load(path_to_model))
    # model = DummySemanticProbabilityModel()
    # model.eval()
    models = [f"/home/mwynen/Dataset321_WMLIS/SEMANTIC_lr1e-2_e2500_checkpoints/fold_{f}/checkpoint_best_Dice_Score_CC.pth"
              for f in range(5)]


    p = SemanticPredictor(
        plans_manager=plans_manager,
        model=models,
        postprocessor=ACLSPostprocessor(
            minimum_instance_size=3,
            semantic_threshold=0.5,
        ),
        output_dir='/home/mwynen/data/nnUNet/tmp/output_dir_test_semantic',
        # preprocessed_files_dir='/home/mwynen/data/nnUNet/tmp/preprocessed_dir',
        num_workers=1,
        save_only_instance_segmentation=False
    )
    p.predict_from_raw_input_dir(input_dir='/home/mwynen/data/nnUNet/nnUNet_raw/Dataset321_WMLIS/imagesTs')

    # p.predict_from_preprocessed_dir()

    # dataloader = p.get_dataloader()
    # predictions_loader = p.get_predictions_loader(dataloader, p.model)
    #
    # for data_batch, predicted_batch in zip(dataloader, predictions_loader):
    #     break

