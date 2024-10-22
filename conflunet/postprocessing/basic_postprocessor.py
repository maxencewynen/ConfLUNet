import torch
import time
import numpy as np
from scipy.ndimage import label
from typing import Callable, Tuple, Dict
from monai.config.type_definitions import NdarrayOrTensor

from conflunet.training.utils import get_default_device


class Postprocessor(Callable):
    def __init__(
            self,
            minimum_instance_size: int = 0,
            minimum_size_along_axis: int = 0,
            semantic_threshold: float = 0.5,
            voxel_spacing: Tuple[float, float, float] = None,
            name: str = "",
            device: torch.device = None,
            verbose: bool = True
    ):
        super(Postprocessor, self).__init__()
        self.minimum_instance_size = minimum_instance_size
        self.minimum_size_along_axis = minimum_size_along_axis
        self.voxel_spacing = voxel_spacing
        self.device = get_default_device() if device is None else device
        assert 0 <= semantic_threshold <= 1, "Threshold should be between 0 and 1"
        self.semantic_threshold = semantic_threshold
        self.name = name
        self.verbose = verbose

    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _maybe_convert_to_tensor(self, data: NdarrayOrTensor) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data
        else:
            return torch.tensor(data).to(self.device)

    @staticmethod
    def _maybe_convert_to_numpy(data: NdarrayOrTensor) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return data
        else:
            return data.cpu().numpy()

    def _convert_as(self, data: NdarrayOrTensor, reference: NdarrayOrTensor) -> NdarrayOrTensor:
        if isinstance(reference, torch.Tensor):
            return self._maybe_convert_to_tensor(data)
        elif isinstance(reference, np.ndarray):
            return self._maybe_convert_to_numpy(data)
        else:
            raise ValueError("reference must be a torch.Tensor or np.ndarray")

    def binarize_semantic_probability(self, semantic_pred_proba: NdarrayOrTensor) -> NdarrayOrTensor:
        binary_pred = (semantic_pred_proba > self.semantic_threshold)
        if isinstance(semantic_pred_proba, torch.Tensor):
            return binary_pred.to(torch.int16)
        else:
            return binary_pred.astype(np.int16)

    def is_too_small(
            self,
            instance_segmentation: np.array,
            instance_id: int,
    ) -> bool:
        """
            Check if a instance is too small to be considered a real instance.
            Args:
                instance_segmentation (np.array): The instance mask.
                instance_id (int): The id of the instance to be checked.
        """
        this_instance_indices = np.where(instance_segmentation == instance_id)
        if len(this_instance_indices[0]) == 0:
            return True
        size_along_x = (1 + max(this_instance_indices[0]) - min(this_instance_indices[0])) * self.voxel_spacing[0]
        size_along_y = (1 + max(this_instance_indices[1]) - min(this_instance_indices[1])) * self.voxel_spacing[1]
        if len(this_instance_indices) == 3:
            size_along_z = (1 + max(this_instance_indices[2]) - min(this_instance_indices[2])) * self.voxel_spacing[2]
            # if the connected component is smaller than 3mm in any direction, skip it as it is not
            # clinically considered a instance
            if (size_along_x < self.minimum_size_along_axis or
                size_along_y < self.minimum_size_along_axis or
                size_along_z < self.minimum_size_along_axis):
                return True

        elif size_along_x < self.minimum_size_along_axis or size_along_y < self.minimum_size_along_axis:
            return True

        return len(this_instance_indices[0]) * np.prod(self.voxel_spacing) <= self.minimum_instance_size

    def remove_small_instances(self, output_dict: Dict[str, NdarrayOrTensor]) -> Dict[str, NdarrayOrTensor]:
        if (self.minimum_instance_size == 0 and self.minimum_size_along_axis == 0) or self.voxel_spacing is None:
            return output_dict

        assert 'instance_seg_pred' in output_dict.keys(), "output_dict must contain 'instance_seg_pred'"
        instance_seg_pred = self._maybe_convert_to_numpy(output_dict['instance_seg_pred'])

        label_list, label_counts = np.unique(instance_seg_pred, return_counts=True)

        instance_seg2 = np.zeros_like(instance_seg_pred)

        for instance_id, lvoxels in zip(label_list, label_counts):
            if instance_id == 0: continue

            if not self.is_too_small(instance_seg_pred, instance_id):
                instance_seg2[instance_seg_pred == instance_id] = instance_id
            elif 'semantic_pred_binary' in output_dict.keys():
                output_dict['semantic_pred_binary'][instance_seg_pred == instance_id] = 0

        output_dict['instance_seg_pred'] = self._convert_as(instance_seg2, output_dict['instance_seg_pred'])
        binary_pred = (instance_seg2 > 0).astype(np.int32)
        output_dict['semantic_pred_binary'] = self._convert_as(binary_pred, output_dict['semantic_pred_binary'])

        return output_dict
    
    def refine_instance_segmentation(self, output_dict: Dict[str, NdarrayOrTensor]) -> Dict[str, NdarrayOrTensor]:
        """
        Refines the instance segmentation by relabeling disconnected components in instances
        and removing ALL instances strictly smaller than the minimum size.
        """    
        self.vprint("[INFO] Refining output instance segmentation...", end=" ")
        assert 'instance_seg_pred' in output_dict.keys(), "output_dict must contain 'instance_seg_pred'"
        instance_mask = self._maybe_convert_to_numpy(output_dict['instance_seg_pred'])
        voxel_size = output_dict['properties']['sitk_stuff']['spacing']
        if voxel_size is None:
            raise ValueError("Voxel spacing must be provided in output_dict")

        iids = np.unique(instance_mask[instance_mask != 0])
        total_iids = len(iids)
        self.vprint(f"({total_iids} predicted instances to review)")
        max_instance_id = np.max(instance_mask)
        resulting_instance_mask = np.copy(instance_mask)

        start = time.time()

        # for every instance id
        for iid in iids:
            # get the mask
            mask = (instance_mask == iid)
            components, n_components = label(mask)
            if n_components == 1: # if the instance is not split in components
                # check if the instance is too small
                if self.is_too_small(components, 1):
                    resulting_instance_mask[mask] = 0
                continue
    
            elif n_components > 1:  # if the instance is split in n components
                # first get the ids and sizes of the different connected components and sort them by size
                this_ids_components_ids, this_ids_components_sizes = np.unique(components[components != 0], return_counts=True)
                sorted_indices = np.argsort(-this_ids_components_sizes)
                sorted_this_ids_components_ids = this_ids_components_ids[sorted_indices]
                sorted_this_ids_components_sizes = this_ids_components_sizes[sorted_indices]
    
                previous_component_is_too_small = False
                # iterate through the different connected components of the instance
                for j, (cid, csize) in enumerate(zip(sorted_this_ids_components_ids, sorted_this_ids_components_sizes)):
                    if previous_component_is_too_small: # since the components are sorted by size, if the previous one is too small, all the others are too
                        resulting_instance_mask[components == cid] = 0
                        continue
    
                    this_component_is_too_small = self.is_too_small(components, cid)
    
                    # if the component is too small, and it is the first one, we remove the whole instance
                    if j == 0 and this_component_is_too_small:
                        resulting_instance_mask[mask] = 0
                        break
    
                    # if the component is too small and it is not the first one, we remove it
                    if this_component_is_too_small:
                        resulting_instance_mask[components == cid] = 0
                        previous_component_is_too_small = True
                        continue
    
                    # if j == 0 it means that the first component, associated with iid, is large enough and we do not need to do change the id
                    # the component is large enough, but basically no center was found in it
                    # In this case, we choose to keep it
                    if j != 0:
                        resulting_instance_mask[components == cid] = max_instance_id + 1
                        max_instance_id += 1

        # Re-identify instance ids to make them contiguous
        dtype = resulting_instance_mask.dtype
        unique_values = np.unique(resulting_instance_mask)
        # Create a mapping from the unique values to contiguous integers (0, 1, 2, ...)
        mapping = {old_value: new_value for new_value, old_value in enumerate(unique_values)}
        # Apply the mapping to the instance mask
        resulting_instance_mask = np.vectorize(mapping.get)(resulting_instance_mask).astype(dtype)

        # Save in output_dict
        output_dict["semantic_pred_binary"] = self._convert_as((resulting_instance_mask > 0).astype(np.int16),
                                                               output_dict["semantic_pred_binary"])
        output_dict['instance_seg_pred'] = self._convert_as(resulting_instance_mask, output_dict['instance_seg_pred'])

        self.vprint(f"       Done. Refinement took: {round(time.time() - start, 1)}s")
        self.vprint(f"       Final number of predicted instances: {len(np.unique(resulting_instance_mask)) - 1}.")

        return output_dict

    def _postprocess(self, output_dict: Dict[str, NdarrayOrTensor]) -> Dict[str, NdarrayOrTensor]:
        return output_dict

    def __call__(self, output_dict: Dict[str, NdarrayOrTensor]) -> Dict[str, NdarrayOrTensor]:
        self.vprint("[INFO] Postprocessing... (postprocessor: {})".format(self.__class__.__name__))
        return self._postprocess(output_dict)
