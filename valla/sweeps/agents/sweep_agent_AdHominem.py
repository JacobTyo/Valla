import wandb
import argparse
import logging
import functools

from valla.methods.AdHominem import sweep_adhominem

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    # get command line args
    parser = argparse.ArgumentParser(description='sweep for bertaa')

    parser.add_argument('--sweep_id', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_path', type=str)

    args = parser.parse_args()

    sweep_fn = functools.partial(sweep_adhominem, project=args.project, device=args.device, dataset=args.dataset,
                                 data_path=args.data_path)

    wandb.agent(args.sweep_id, sweep_fn, project=args.project)
