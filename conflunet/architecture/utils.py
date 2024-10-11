import torch
from conflunet.utilities.planning_and_configuration import ConfigurationManagerInstanceSeg
from conflunet.architecture.nnconflunet import get_network_from_plans


def get_model(configuration: ConfigurationManagerInstanceSeg, n_channels: int, semantic: bool = False):
    _model = configuration.network_arch_class_name if semantic else "conflunet.architecture.nnconflunet.nnConfLUNet"
    return get_network_from_plans(
        _model,
        configuration.network_arch_init_kwargs,
        configuration.network_arch_init_kwargs_req_import,
        n_channels,
        output_channels=2,
        allow_init=True,
        deep_supervision=False
    )


def load_model(configuration: ConfigurationManagerInstanceSeg, checkpoint: str, n_channels: int, semantic: bool = False):
    model = get_model(configuration, n_channels, semantic)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    return model
