import argparse


def main(args):
    from conflunet.evaluation.metrics import compute_and_save_metrics_from_model_and_postprocessor
    compute_and_save_metrics_from_model_and_postprocessor(args.dataset_name, args.model_name, args.postprocessor_name, args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation from model name and postprocessor name')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--postprocessor_name', type=str)
    parser.add_argument('--save_dir', type=str, help='Path to the output directory')

    args = parser.parse_args()

    main(args)