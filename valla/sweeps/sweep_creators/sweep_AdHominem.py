import wandb
import argparse
import logging
import math

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    # get command line args
    parser = argparse.ArgumentParser(description='adhom sweep config')

    parser.add_argument('--project', type=str)

    args = parser.parse_args()

    wandb.login()

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'auc',
            'goal': 'maximize'
        },
        'parameters': {
            'epochs': {
                'value': 20
            },
            'vocab_sizes': {
                'values': [
                    [1000, 150],
                    [5000, 150],
                    [10000, 150],
                    [20000, 150],
                    [30000, 150],
                    [40000, 150],
                    [50000, 150],
                    [5000, 100],
                    [5000, 200],
                    [5000, 250],
                    [5000, 500],
                    [5000, 1000],
                    [5000, 2000],
                ]
            },
            'D_c': {  # character embeding dimension
                'distribution': 'q_log_uniform',
                'min': math.log(2),
                'max': math.log(1024),
            },
            'D_r': {  # character representation dimension
                'distribution': 'q_log_uniform',
                'min': math.log(2),
                'max': math.log(1024),
            },
            'w': {
                'values': [2, 4, 6, 8],
            },
            'D_w': {
                'values': [300]
            },
            'D_s': {
                'distribution': 'q_log_uniform',
                'min': math.log(2),
                'max': math.log(128),
            },
            'D_d': {
                'distribution': 'q_log_uniform',
                'min': math.log(2),
                'max': math.log(128),
            },
            'D_mlp': {
                'distribution': 'q_log_uniform',
                'min': math.log(2),
                'max': math.log(256),
            },
            'T_c': {
                'distribution': 'q_log_uniform',
                'min': math.log(1),
                'max': math.log(20),
            },
            'T_w': {
                'values': [30]
            },
            'train_word_embeddings': {
                'values': [True, False]
            },
            'T_s': {
                'distribution': 'q_log_uniform',
                'min': math.log(1),
                'max': math.log(512),
            },
            'batch_size_tr': {
                'values': [32]  # , 48, 64, 80]
            },
            'batch_size_te': {
                'values': [128]
            },
            'initial_learning_rate': {
                'distribution': 'log_uniform',
                'min': math.log(0.00001),
                'max': math.log(0.01),
            },
            'keep_prob_cnn': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 1,
            },
            'keep_prob_lstm': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 1,
            },
            'keep_prob_att': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 1,
            },
            'keep_prob_metric': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 1,
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'eta': 3,
            'max_iter': 50
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=args.project)
    # logging.info(f'sweep id: {sweep_id}')
