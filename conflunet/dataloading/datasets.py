import monai
from typing import List, Union
from monai.data import CacheDataset, DataLoader

from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.training.dataloading.utils import get_case_identifiers


class LesionInstancesDataset(CacheDataset):
    def __init__(self, folder: str,
                 case_identifiers: Union[List[str], None] = None,
                 transforms: monai.transforms = None,
                 cache_rate: float = 1.0,
                 num_workers: int = 0):
        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()

        dataset = []
        for c in case_identifiers:
            dataset.append({'name': c,
                            'data': join(folder, f"{c}.npz"),
                            'properties_file': join(folder, f"{c}.pkl")})

        self.dataset = dataset
        super().__init__(data=dataset, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)

    def make_dataloader(self, batch_size=1, num_workers=0, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)