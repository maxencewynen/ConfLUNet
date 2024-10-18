import torch
import time
from typing import Optional, Dict

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
            save_only_instance_segmentation: bool = True
    ):
        super(ConfLUNetPredictor, self).__init__(
            plans_manager=plans_manager,
            model=model,
            postprocessor=postprocessor,
            output_dir=output_dir,
            preprocessed_files_dir=preprocessed_files_dir,
            num_workers=num_workers,
            save_only_instance_segmentation=save_only_instance_segmentation
        )

    def predict_batch(self, batch: dict, model: torch.nn.Module = None) -> Dict[str, NdarrayOrTensor]:
        if model is not None:
            self.model = model
        assert self.model is not None, "Model must be provided"
        device = self.device

        self.model.to(device)

        img = batch["img"].to(device)
        assert img.shape[0] == 1, "Batch size for original batch must be 1"

        spatial_shape = img.shape[-3:]
        brainmask = batch["brainmask"].to(device) if "brainmask" in batch.keys() else torch.ones(spatial_shape).to(
            device)

        patch_size = self.patch_size
        start = time.time()
        with torch.no_grad():
            outputs = sliding_window_inference(img, patch_size, self.batch_size, self.model, mode='gaussian', overlap=0.5)
        print(f"[INFO] Sliding window inference took {time.time() - start:.2f} seconds")
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
    from conflunet.architecture.conflunet import ConfLUNet
    from torch import nn
    import torch
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(321)

    model = ConfLUNet(1,2, scale_offsets=20)
    path_to_model = "/home/mwynen/Downloads/best_DSC_SO_A_L1_e-5_h1200o0.3_S20_seed1.pth"
    model.load_state_dict(torch.load(path_to_model))
    # model = UNet3D(1,2)
    # path_to_model = "/home/mwynen/Downloads/best_DSC_SS_lre-5_seed1.pth"
    # model.load_state_dict(torch.load(path_to_model))
    # model = DummySemanticProbabilityModel()
    model.eval()

    p = ConfLUNetPredictor(
        plans_manager=plans_manager,
        model=model,
        postprocessor=ConfLUNetPostprocessor(
            minimum_instance_size=3,
            semantic_threshold=0.5,
        ),
        output_dir='/home/mwynen/data/nnUNet/tmp/output_dir_test',
        preprocessed_files_dir='/home/mwynen/data/nnUNet/tmp/preprocessed_dir',
        num_workers=0,
        save_only_instance_segmentation=False
    )
    p.predict_from_preprocessed_dir()
    pass
