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
    model = get_network_from_plans(
        _model,
        configuration.network_arch_init_kwargs,
        configuration.network_arch_init_kwargs_req_import,
        n_channels,
        output_channels=2,
        allow_init=True,
        deep_supervision=False
    ).to(device)

    # Initialize optimizer and scheduler according to nnunet
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, weight_decay=3e-5, momentum=0.99)
    lr_scheduler = PolyLRScheduler(optimizer, args.learning_rate, args.n_epochs)
    start_epoch = 0

    if os.path.exists(checkpoint_filename) and not args.force_restart:
        checkpoint = torch.load(checkpoint_filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint["scheduler"])

        if not args.wandb_ignore:
            wandb_run_id = checkpoint['wandb_run_id']
            wandb.init(project=args.wandb_project, mode="online", name=args.name, resume="must", id=wandb_run_id)
        print(f"\nResuming training: (epoch {checkpoint['epoch']})\nLoaded checkpoint '{checkpoint_filename}'\n")

    elif not args.wandb_ignore:
        wandb.login()
        wandb.init(project=args.wandb_project, mode="online", name=args.name)
        wandb_run_id = wandb.run.id

    return model, optimizer, lr_scheduler, start_epoch, wandb_run_id





