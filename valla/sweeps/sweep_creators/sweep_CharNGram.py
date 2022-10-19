import wandb
import argparse
import logging
import math

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of the blogs dataset')

    parser.add_argument('--sweep_name', type=str)
    parser.add_argument('--project', type=str)

    args = parser.parse_args()

    wandb.login()

    sweep_config = {
        'name': args.sweep_name,
        'method': 'random',
        'metric': {
            'name': 'val/macro_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'gram_range': {
                'values': [[1, 3],
                           [2, 4],
                           [3, 5]]
            },
            'n_best_factor': {
                'values': [1],
                # 'distribution': 'uniform',
                # 'min': 0.5,
                # 'max': 1
            },
            'use_lsa': {
                'values': [False]
            },
            'lsa_factors': {
                'values': [63]
                # 'distribution': 'int_uniform',
                # 'min': 8,
                # 'max': 256
            },
            'max_features': {
                # log uniforms are really nice - try out orders of magnitude
                'distribution': 'int_uniform',
                'min': 10_000,
                'max': 150_000
            },
            'min_df': {
                'distribution': 'uniform',
                'min': 0,
                'max': 0.1,
            },
            'sublinear_tf': {
                'values': [True, False]
            },
            'primal': {
                'values': [False]
            },
            'logistic_regression': {
                'values': [True, False]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=args.project)
