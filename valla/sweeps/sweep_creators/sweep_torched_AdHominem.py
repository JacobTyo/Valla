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
        'method': 'bayes',
        'metric': {
            'name': 'macro_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'logging_steps': {'value': 2000},
            'save_best_model': {'value': True},
            'save_model_checkpoints': {'value': False},
            'save_model_dir': {'value': 'adhominem_sweep_models'},
            'num_dataloader_workers': {'value': 32},
            'epochs': {'value': 50},
            'train_batch_size': {'value': 40},
            'lr': {
                'distribution': 'log_uniform',
                'min': math.log(0.00001),
                'max': math.log(0.1),
            },
            'weight_decay': {
                'distribution': 'log_uniform',
                'min': math.log(0.00001),
                'max': math.log(0.1),
            },
            'loss_margin': {'value': 0.05},
            'cnn_stride': {
                'values': [1]
            },
            'D_c': {  # character embeding dimension
                'distribution': 'q_log_uniform',
                'min': math.log(32),
                'max': math.log(1024),
            },
            'D_r': {  # character representation dimension
                'distribution': 'q_log_uniform',
                'min': math.log(32),
                'max': math.log(1024),
            },
            'w': {
                'values': [4],
            },
            'D_w': {
                'values': [300]
            },
            'D_s': {
                'distribution': 'q_log_uniform',
                'min': math.log(32),
                'max': math.log(128),
            },
            'D_d': {
                'distribution': 'q_log_uniform',
                'min': math.log(32),
                'max': math.log(128),
            },
            'D_mlp': {
                'distribution': 'q_log_uniform',
                'min': math.log(32),
                'max': math.log(256),
            },
            'max_chars_per_word': {
                'distribution': 'q_log_uniform',
                'min': math.log(1),
                'max': math.log(20),
            },
            'max_words_per_sentence': {
                'distribution': 'q_log_uniform',
                'min': math.log(1),
                'max': math.log(50),
            },
            'max_sentences_per_doc': {
                'distribution': 'q_log_uniform',
                'min': math.log(1),
                'max': math.log(256),
            },
            'cnn_dropout_prob': {
                'distribution': 'log_uniform',
                'min': math.log(0.00001),
                'max': math.log(0.5),
            },
            'w2s_dropout_prob': {
                'distribution': 'log_uniform',
                'min': math.log(0.00001),
                'max': math.log(0.5),
            },
            'w2s_att_dropout_prob': {
                'distribution': 'log_uniform',
                'min': math.log(0.00001),
                'max': math.log(0.5),
            },
            's2d_dropout_prob': {
                'distribution': 'log_uniform',
                'min': math.log(0.00001),
                'max': math.log(0.5),
            },
            's2d_att_dropout_prob': {
                'distribution': 'log_uniform',
                'min': math.log(0.00001),
                'max': math.log(0.5),
            },
            'metric_dropout_prob': {
                'distribution': 'log_uniform',
                'min': math.log(0.00001),
                'max': math.log(0.5),
            },
            'chr_vocab_size': {
                'distribution': 'q_log_uniform',
                'min': math.log(10),
                'max': math.log(512),
            },
            'tok_vocab_size': {
                'distribution': 'q_log_uniform',
                'min': math.log(100),
                'max': math.log(50000),
            },
            'dont_use_fasttext': {'value': False},
            'max_grad_norm': {
                'distribution': 'log_uniform',
                'min': math.log(0.1),
                'max': math.log(10),
            },
            'lr_decay_gamma': {
                'distribution': 'log_uniform',
                'min': math.log(0.8),
                'max': math.log(0.96),
            },
            'chr_count_min': {
                'distribution': 'q_log_uniform',
                'min': math.log(1),
                'max': math.log(1000),
            },
            'tok_count_min': {
                'distribution': 'q_log_uniform',
                'min': math.log(1),
                'max': math.log(1000),
            },
            'evaluate_every_epoch': {'value': True},
            'evaluation_steps': {'value': -1},
            'test_batch_size': {'value': 40},
            'AA': {'value': False}
        },
        'early_terminate': {
            'type': 'hyperband',
            'eta': 2,
            'min_iter': 2
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=args.project)
