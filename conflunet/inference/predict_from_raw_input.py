# Runs inference of a single model on input files.
# Options: To perform ensemble inference / To perform inference with a single model.
import argparse
import torch
import warnings
from typing import Union, List
from os.path import exists as pexists
from os.path import join as pjoin

from nnunetv2.paths import nnUNet_results

from conflunet.postprocessing.instance import ConfLUNetPostprocessor
from conflunet.inference.predictors.instance import ConfLUNetPredictor
from conflunet.inference.predictors.semantic import SemanticPredictor
from conflunet.postprocessing.semantic import ACLSPostprocessor, ConnectedComponentsPostprocessor
from conflunet.utilities.planning_and_configuration import load_dataset_and_configuration


def predict_from_raw_input_and_model_name(
        dataset_id: int,
        input_dir: str,
        output_dir: str,
        model_name: str,
        semantic: bool = False,
        postprocessor: str = 'ConfLUNet',
        postprocessor_kwargs: dict = None,
        num_workers: int = 4,
        verbose: bool = False,
) -> None:
    dataset_name, plans_manager, configuration, n_channels = load_dataset_and_configuration(dataset_id)
    if torch.cuda.is_available():
        device = torch.cuda
    else:
        warnings.warn("CUDA is not available. Running on CPU, but this will be slow.")
        device = torch.cpu

    base_dir = pjoin(nnUNet_results, dataset_name, model_name)
    if not semantic and postprocessor == 'ConfLUNet':
        checkpoint_name = "checkpoint_best_Panoptic_Quality_ConfLUNet.pth"
    else:
        checkpoint_name = f"checkpoint_best_Dice_Score_{postprocessor}.pth"

    postprocessor_class = ConfLUNetPostprocessor if not semantic else None
    if postprocessor_class is None:
        postprocessor_class = ACLSPostprocessor if postprocessor == 'ACLS' else ConnectedComponentsPostprocessor

    predictor_class = ConfLUNetPredictor if not semantic else SemanticPredictor
    model = [pjoin(base_dir, f"fold_{i}", checkpoint_name) for i in range(5)]
    if not pexists(model[0]):
        raise FileNotFoundError(f"Model checkpoint not found at {model[0]}")

    if postprocessor_kwargs is None:
        postprocessor_kwargs = {}

    predictor = predictor_class(
        plans_manager=plans_manager,
        model=model,
        postprocessor=postprocessor_class(**postprocessor_kwargs),
        output_dir=output_dir,
        num_workers=num_workers,
        verbose=verbose
    )

    predictor.predict_from_raw_input_dir(input_dir)


def predict_from_raw_input_dir(
        dataset_id: int,
        input_dir: str,
        output_dir: str,
        model: Union[str, List[str]],
        semantic: bool = False,
        postprocessor: str = 'ConfLUNet',
        postprocessor_kwargs: dict = None,
        num_workers: int = 4,
        verbose: bool = False,
) -> None:
    """
    model can be a single model or a list of models. If it is a list of models, it will perform ensemble inference.
    it can either be a path or a model name. In the latter case, all 5 models will be loaded from nnUNet_results
    """
    if isinstance(model, str) and not pexists(model): # if model is a model name
        return predict_from_raw_input_and_model_name(
            dataset_id=dataset_id,
            input_dir=input_dir,
            output_dir=output_dir,
            model_name=model,
            semantic=semantic,
            postprocessor=postprocessor,
            postprocessor_kwargs=postprocessor_kwargs,
            num_workers=num_workers,
            verbose=verbose
        )

    else:
        raise NotImplementedError()


def main(args):
    postprocessor_kwargs = {}
    for arg in ["minimum_instance_size", "minimum_size_along_axis", "semantic_threshold", "heatmap_threshold",
                "nms_kernel_size", "top_k"]:
        if getattr(args, arg) is not None:
            postprocessor_kwargs[arg] = getattr(args, arg)

    predict_from_raw_input_dir(
        dataset_id=args.dataset_id,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=args.model,
        semantic=args.semantic,
        postprocessor=args.postprocessor,
        postprocessor_kwargs=postprocessor_kwargs,
        num_workers=args.num_workers,
        verbose=args.verbose
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--semantic", action='store_true')
    parser.add_argument("--postprocessor", type=str, default='ConfLUNet')
    parser.add_argument("--minimum_instance_size", type=float, default=None)
    parser.add_argument("--minimum_size_along_axis", type=float, default=None)
    parser.add_argument("--semantic_threshold", type=float, default=None)
    parser.add_argument("--heatmap_threshold", type=float, default=None)
    parser.add_argument("--nms_kernel_size", type=float, default=None)
    parser.add_argument("--top_k", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    main(args)




