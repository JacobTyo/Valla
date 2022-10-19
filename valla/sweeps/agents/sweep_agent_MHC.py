import wandb
import argparse
import logging
import functools

from valla.methods.AA_MHC import run_mhc_sweep

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    # get command line args
    parser = argparse.ArgumentParser(description='sweep for aa-mhc')

    parser.add_argument('--sweep_id', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--val_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--cache_dir', type=str, default='mhc_cache')
    parser.add_argument('--save_dir', type=str, default='mhc_save')
    parser.add_argument('--device', type=int)

    args = parser.parse_args()

    sweep_fn = functools.partial(run_mhc_sweep, train_pth=args.train_path, val_path=args.val_path,
                                 test_pth=args.test_path, device=args.device, project=args.project,
                                 cache_dir=args.cache_dir, save_path=args.save_dir)

    wandb.agent(args.sweep_id, sweep_fn, project=args.project)
