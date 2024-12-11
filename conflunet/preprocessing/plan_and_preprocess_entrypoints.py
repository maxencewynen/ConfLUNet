import argparse
from conflunet.preprocessing.preprocess import *


def extract_fingerprint_entry():
    from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import extract_fingerprint_entry
    extract_fingerprint_entry()


def plan_experiment_entry():
    from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_experiment_entry
    plan_experiment_entry()


def preprocess_entry():
    parser = argparse.ArgumentParser(description="Preprocess dataset.")
    parser.add_argument("--dataset_id", required=True, type=int, help="ID of the dataset to preprocess.")
    parser.add_argument("--check_dataset_integrity", action="store_true", help="Check dataset integrity.")
    parser.add_argument("--num_processes", type=int, default=default_num_processes, help="Number of processes to use.")
    parser.add_argument("--overwrite_existing_dataset_fingerprint", action="store_true",
                        help="Overwrite existing dataset fingerprint.")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--output_dir_for_inference", type=str,
                        help="if inference only, path to where to store the temporary preprocessed files.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    preprocess(args.dataset_id, args.check_dataset_integrity, args.num_processes,
               args.overwrite_existing_dataset_fingerprint,
               inference=args.inference, output_dir_for_inference=args.output_dir_for_inference, verbose=args.verbose)


def plan_and_preprocess_entry():
    preprocess_entry()
