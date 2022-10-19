import wandb
import argparse
import logging
import math

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of the blogs dataset')

    parser.add_argument('--sweep_name', type=str)

    args = parser.parse_args()

    wandb.login()

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'test_U/macro_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'epochs': {
                'value': 20
            },
            'vocab_size': {
                'distribution': 'int_uniform',
                'min': 32,
                'max': 4096
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'eta': 3,
            'max_iter': 20
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=args.sweep_name)
    logging.info(f'sweep id: {sweep_id}')
