import wandb
import argparse
import logging
import functools

from valla.methods.PPM import run_ppm

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    # get command line args
    parser = argparse.ArgumentParser(description='sweep for n-gram models')

    parser.add_argument('--sweep_id', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--val_path', type=str)

    args = parser.parse_args()

    sweep_fn = functools.partial(run_ppm, train_pth=args.train_path, test_pth=args.val_path)
    wandb.agent(args.sweep_id, sweep_fn, project=args.project)
