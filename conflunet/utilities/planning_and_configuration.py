from os.path import join as pjoin
from functools import lru_cache

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name, convert_dataset_name_to_id

from conflunet.preprocessing.preprocessors import InstanceSegProcessor


class PlansManagerInstanceSeg(PlansManager):
    def __init__(self, plans_file: str):
        super(PlansManagerInstanceSeg, self).__init__(plans_file)

    @lru_cache(maxsize=10)
    def get_configuration(self, configuration_name: str):
        configuration_dict = self._internal_resolve_configuration_inheritance(configuration_name)
        return ConfigurationManagerInstanceSeg(configuration_dict)

    @property
    def n_channels(self):
        return len(self.foreground_intensity_properties_per_channel)

    @property
    def dataset_id(self):
        return convert_dataset_name_to_id(self.dataset_name)


class ConfigurationManagerInstanceSeg(ConfigurationManager):
    def __init__(self, configuration_dict):
        super(ConfigurationManagerInstanceSeg, self).__init__(configuration_dict)

    @property
    @lru_cache(maxsize=1)
    def preprocessor_class(self):
        return InstanceSegProcessor


def load_dataset_and_configuration(dataset_id: int):
    dataset_name = convert_id_to_dataset_name(dataset_id)
    plans_file = pjoin(nnUNet_preprocessed, dataset_name, 'nnUNetPlans.json')
    plans_manager = PlansManagerInstanceSeg(plans_file)
    configuration = plans_manager.get_configuration('3d_fullres')
    n_channels = len(plans_manager.foreground_intensity_properties_per_channel)

    return dataset_name, plans_manager, configuration, n_channels