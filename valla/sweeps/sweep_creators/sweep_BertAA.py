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
            'name': 'macro_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'epochs': {
                'value': [5]
            },
            'model_path': {
                'values': ['bert-base-cased', 'vinai/bertweet-base']  #, 'vinai/bertweet-large']
            },
            'optimizer': {
                'values': ['AdamW'],  # , 'Adafactor'], # causing issues
            },
            'learning_rate': {
                # log uniforms are really nice - try out orders of magnitude
                'distribution': 'log_uniform',
                'min': math.log(1e-6),
                'max': math.log(0.001),
            },
            'warmup_steps': {
                # 'values': [0.15]
                'distribution': 'q_uniform',
                'min': 0,
                'max': 10000
            },
            'weight_decay': {
                # log uniforms are really nice - try out orders of magnitude
                'distribution': 'log_uniform',
                'min': math.log(1e-8),
                'max': math.log(0.1),
            },
            'batch_size': {
                'values': [84]
                # 'values': [48]
                # 'values': [8, 12, 16],
            },
            'max_seq_len': {
                'values': [128, 512]  # 64
            },
            'doc_stride': {
                'values': [0.8]
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'eta': 3,
            'max_iter': 100
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=args.project)
    # logging.info(f'sweep id: {sweep_id}')  # this is logged to the console anyway
