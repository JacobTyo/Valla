import wandb
import argparse
import logging
import functools

from valla.methods.CharNGram import run_ngram

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    # get command line args
    parser = argparse.ArgumentParser(description='sweep for n-gram models')

    parser.add_argument('--sweep_id', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--val_path', type=str)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--device', type=int)

    args = parser.parse_args()

    sweep_fn = functools.partial(run_ngram, ngram_type=args.model, train_pth=args.train_path, test_pth=args.val_path,
                                 project=args.project, num_workers=args.num_workers)
    wandb.agent(args.sweep_id, sweep_fn, project=args.project)
