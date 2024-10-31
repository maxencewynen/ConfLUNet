import argparse
import warnings

from conflunet.training.semantic_trainer import SemanticTrainer
from conflunet.training.conflunet_trainer import ConfLUNetTrainer
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get all command line arguments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    parser.add_argument('--dataset_id', type=int, default=321, help='Specify the dataset id')
    parser.add_argument('--fold', type=int, default=0, help='Specify the fold')
    parser.add_argument('--semantic', default=False, action='store_true', help='Whether to train the semantic segmentation model')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')
    parser.add_argument('--cache_rate', default=0, type=float)
    parser.add_argument('--dice_loss_weight', type=float, default=0.5, help='Specify the weight of the dice loss')
    parser.add_argument('--focal_loss_weight', type=float, default=1, help='Specify the weight of the focal loss')
    parser.add_argument('--seg_loss_weight', type=float, default=1, help='Specify the weight of the segmentation loss')
    parser.add_argument('--heatmap_loss_weight', type=float, default=1, help='Specify the weight of the heatmap loss')
    parser.add_argument('--offsets_loss_weight', type=float, default=1, help='Specify the weight of the offsets loss')
    parser.add_argument('--offsets_loss', type=str, default="l1", help="Specify the loss used for the offsets. ('sl1' or 'l1')")
    parser.add_argument('--get_small_instances', default=False, action='store_true', help="Whether to use a weighted loss for small instances")
    parser.add_argument('--get_confluent_instances', default=False, action='store_true', help="Whether to use a weighted loss for confluent instances")
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Specify the initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=1500, help='Specify the number of epochs to train for')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation every n-th epochs (loss only, patches only)')
    parser.add_argument('--actual_val_interval', type=int, default=50, help='Full validation with metrics computation every n-th epochs (full images, inference with sliding windows')
    parser.add_argument('--wandb_ignore', action="store_true", default=False, help='Whether to ignore this run in wandb')
    parser.add_argument('--model_name', default="idiot without a name", help='Wandb run name')
    parser.add_argument('--force_restart', default=False, action='store_true', help="force the training to restart at 0 even if a checkpoint was found")
    parser.add_argument('--debug', default=False, action='store_true', help="Debug mode: use a single batch for training and validation")
    parser.add_argument('--save_predictions', default=False, action='store_true', help="Debug mode: save predictions of the first batch for each validation step")

    args = parser.parse_args()
    if args.semantic:
        kwargs = vars(args)
        del kwargs['heatmap_loss_weight'], kwargs['offsets_loss_weight'], kwargs['offsets_loss'], kwargs['semantic'], kwargs['seg_loss_weight']
        kwargs['wandb_project'] = "ConfLUNet_Semantic"
        trainer = SemanticTrainer(**vars(args))
    else:
        kwargs = vars(args)
        del kwargs['semantic']
        kwargs['wandb_project'] = "ConfLUNet"
        trainer = ConfLUNetTrainer(**vars(args))

    trainer.run_training()
