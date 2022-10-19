import wandb
import argparse
import logging

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
            'name': 'val/macro_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'order': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 5
            },
            'alph_size': {
                'values': [32, 64, 128, 256]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=args.sweep_name)
    logging.info(f'sweep id: {sweep_id}')
