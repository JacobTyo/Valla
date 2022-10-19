import wandb
import argparse
import logging
import functools

from valla.methods.torched_AdHominem import torched_adhom_sweep

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    # get command line args
    parser = argparse.ArgumentParser(description='sweep for bertaa')

    parser.add_argument('--sweep_id', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--device', type=int)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)

    args = parser.parse_args()

    sweep_fn = functools.partial(torched_adhom_sweep, wandb_project=args.project, device=args.device,
                                 train_path=args.train_path, test_path=args.test_path)

    wandb.agent(args.sweep_id, sweep_fn, project=args.project)
