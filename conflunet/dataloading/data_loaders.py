from builtins import bool
from typing import Union, Tuple

from batchgenerators.dataloading.data_loader import DataLoader
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
# from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from conflunet.dataloading.datasets import nnUNetInstancesDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase

from conflunet.preprocessing.preprocess import PlansManagerInstanceSeg


class nnUNetInstancesDataLoaderBase(nnUNetDataLoaderBase):
    # adapted from nnunetv2.training.dataloading.base_data_loader to handle instance segmentation maps
    def __init__(self,
                 data: nnUNetInstancesDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False):
        print("Hello")
        super().__init__(data=data, batch_size=batch_size, patch_size=patch_size, final_patch_size=final_patch_size,
                         oversample_foreground_percent=oversample_foreground_percent, sampling_probabilities=sampling_probabilities,
                         pad_sides=pad_sides, probabilistic_oversampling=probabilistic_oversampling, label_manager=label_manager)

        print()

        self.data_shape, self.seg_shape, self.instance_seg_shape = self._determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple(label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling

    def _determine_shapes(self):
        # load one case
        data, seg, instance_seg_shape, properties = self._data.load_case(self.indices[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        instance_seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        return data_shape, seg_shape, instance_seg_shape

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    print('Warning! No annotated pixels in image!')
                    selected_class = None
                # print(f'I have ignore labels and want to pick a labeled area. annotated_classes_key: {self.annotated_classes_key}')
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in
                       eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (
                                overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs


if __name__ == '__main__':
    pass
    plans_manager = PlansManagerInstanceSeg("/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans.json")
    configuration_manager = plans_manager.get_configuration("3d_fullres")

    dataset_json = "/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/dataset.json"
    dataset_json = load_json(dataset_json)
    label_manager = plans_manager.get_label_manager(dataset_json)

    folder = '/home/mwynen/data/nnUNet/nnUNet_preprocessed/Dataset321_WMLIS/nnUNetPlans_3d_fullres'
    test = nnUNetInstancesDataLoaderBase(data=nnUNetInstancesDataset(folder),
                                         batch_size=2,
                                         patch_size=np.array([64, 64, 64]),
                                         final_patch_size=np.array([64, 64, 64]),
                                         label_manager=label_manager,
                                         oversample_foreground_percent=0.0,
                                         sampling_probabilities=None,
                                         pad_sides=None,
                                         probabilistic_oversampling=False)

    print(test.data_shape)
