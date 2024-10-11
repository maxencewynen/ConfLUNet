import torch
import os
import numpy as np
import random
import nibabel as nib
from os.path import join as pjoin
from monai.utils import set_determinism
import wandb
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from conflunet.architecture.nnconflunet import get_network_from_plans


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        return torch.device('cuda')
    raise Exception("No GPU device was found: I cannot train without GPU.")


def seed_everything(seed_val: int):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    set_determinism(seed=seed_val)


def save_patch(data, name, save_dir):
    tobesaved = data if len(data.shape) == 3 else np.squeeze(data[0,0,:,:,:])
    nib.save(nib.Nifti1Image(tobesaved, np.eye(4)),
            pjoin(save_dir, f'{name}.nii.gz'))


def get_model_optimizer_and_scheduler(args, configuration, n_channels, device, checkpoint_filename, semantic: bool = False):
    wandb_run_id = None
    _model = configuration.network_arch_class_name if semantic else "conflunet.architecture.nnconflunet.nnConfLUNet"
    # Initialize model
    if os.path.exists(checkpoint_filename) and not args.force_restart:
        model = get_network_from_plans(
            _model,
            configuration.network_arch_init_kwargs,
            configuration.network_arch_init_kwargs_req_import,
            n_channels,
            output_channels=2,
            allow_init=True,
            deep_supervision=False
        ).to(device)

        checkpoint = torch.load(checkpoint_filename)

        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, weight_decay=3e-5,
                                    momentum=0.99)  # following nnunet
        optimizer.load_state_dict(checkpoint['optimizer'])

        if not args.wandb_ignore:
            wandb_run_id = checkpoint['wandb_run_id']
            wandb.init(project=args.wandb_project, mode="online", name=args.name, resume="must", id=wandb_run_id)

        # Initialize scheduler
        lr_scheduler = PolyLRScheduler(optimizer, args.learning_rate, args.n_epochs)  # following nnunet
        lr_scheduler.load_state_dict(checkpoint["scheduler"])

        print(f"\nResuming training: (epoch {checkpoint['epoch']})\nLoaded checkpoint '{checkpoint_filename}'\n")

    else:
        print(f"Initializing new model with {n_channels} input channels")
        model = get_network_from_plans(
            "conflunet.architecture.nnconflunet.nnConfLUNet",  # configuration.network_arch_class_name,
            configuration.network_arch_init_kwargs,
            configuration.network_arch_init_kwargs_req_import,
            n_channels,
            output_channels=2,
            allow_init=True,
            deep_supervision=False
        )

        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, weight_decay=3e-5,
                                    momentum=0.99)  # following nnunet

        start_epoch = 0
        if not args.wandb_ignore:
            wandb.login()
            wandb.init(project=args.wandb_project, mode="online", name=args.name)
            wandb_run_id = wandb.run.id

        # Initialize scheduler
        lr_scheduler = PolyLRScheduler(optimizer, args.learning_rate, args.n_epochs)  # following nnunet

    return model, optimizer, lr_scheduler, start_epoch, wandb_run_id





