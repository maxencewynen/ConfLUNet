from typing import Union, List, Tuple

import torch
import numpy as np
from nnunetv2.utilities.helpers import dummy_context
from torch import nn, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from conflunet.architecture.model import get_network_from_plans
from conflunet.training.loss import ConfLUNetLoss
from conflunet.dataloading.datasets import LesionInstancesDataset
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels


class ConfLUNetTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False # Disabling first
        self.num_workers = max(1, get_allowed_n_proc_DA())

    def _build_loss(self):
        seg_loss = super()._build_loss()
        loss = ConfLUNetLoss(seg_loss, center_loss_weight=1., offsets_loss_weight=1.)
        return loss

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            )
            self.network = self.network.to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        from conflunet.dataloading.transforms import get_train_transforms, get_val_transforms
        seed = self.fold if isinstance(self.fold, int) else 0
        dataset_tr = LesionInstancesDataset(self.preprocessed_dataset_folder, tr_keys,
                                            transforms=get_train_transforms(seed=seed, patch_size=self.configuration_manager.patch_size),
                                            cache_rate=1.0,
                                            num_workers=self.num_workers)

        dataset_val = LesionInstancesDataset(self.preprocessed_dataset_folder, val_keys,
                                             transforms=get_val_transforms(seed=seed, patch_size=self.configuration_manager.patch_size),
                                             cache_rate=1.0,
                                             num_workers=self.num_workers)

        return dataset_tr, dataset_val

    def get_dataloaders(self):
        tr, val = self.get_tr_and_val_datasets()
        tr_loader = tr.make_dataloader(batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        val_loader = val.make_dataloader(batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return tr_loader, val_loader

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()





def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          use_compressed: bool = False,
                          device: torch.device = torch.device('cuda')):
    nnunet_trainer = ConfLUNetTrainer

    # handle dataset input. If it's an ID we need to convert to int from string
    if dataset_name_or_id.startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                             f'input: {dataset_name_or_id}')

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, unpack_dataset=not use_compressed, device=device)
    return nnunet_trainer



if __name__=='__main__':
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'

    #Test trainer
    trainer = get_trainer_from_args('321', '3d_fullres', fold=0)

    (
        rotation_for_DA,
        do_dummy_2d_data_aug,
        initial_patch_size,
        mirror_axes,
    ) = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

    trainer.run_training()
    # trainer.initialize()
    # model = trainer.network
    #
    # data = torch.randn(1, 2, 96, 96, 96).to('cuda')
    # out = model(data)
    #
    # ref = (torch.randint(low=0, high=2, size = (1, 2, 96, 96, 96)).to('cuda'),
    #        torch.randn(1, 1, 96, 96, 96).to('cuda'),
    #        torch.randn(1, 3, 96, 96, 96).to('cuda'),)
    #
    # x = trainer.loss(out, ref)
    # print(x)
    pass
    