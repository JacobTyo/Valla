from simpletransformers.classification import ClassificationModel
from sklearn import metrics
from valla.utils import eval_metrics
from valla.utils.dataset_utils import dict_dset_to_list, list_dset_to_dict
import wandb
import os
import pandas as pd
import numpy as np
from functools import partial
from valla.dsets.loaders import aa_as_pandas, get_aa_dataset
import argparse
import logging
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (6144, rlimit[1]))

logging.basicConfig(level=logging.INFO)

DEFAULT_TRAINING_ARGS = {
        'reprocess_input_data': True,
        'num_train_epochs': 5,
        'evaluate_during_training': True,
        "use_early_stopping": True,
        "early_stopping_consider_epochs": True,
        "early_stopping_delta": 0,
        "early_stopping_metric": 'macro_accuracy',
        "early_stopping_metric_minimize": False,
        "early_stopping_patience": 5,
        'evaluate_during_training_steps': 1000,
        'evaluate_each_epoch': True,
        'train_batch_size': 16,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "save_eval_checkpoints": False,
        "save_steps": 1000,
        "save_model_every_epoch": False,
        "save_best_model": False,
        "best_model_dir": 'best_model',
        "output_dir": 'BertAA_output',
        "overwrite_output_dir": True,
        "learning_rate": 3e-5,
        "max_seq_length": 128,
        "sliding_window": False,
        "stride": int(0.8 * 128),
        "logging_steps": 1,
        "warmup_ratio": 0.01,
        "warmup_steps": 1000,
        "weight_decay": 1e-8,
        "manual_seed": 0,
        "wandb_project": 'testing',
        "lazy_loading": False,
        "lazy_labels_column": 1,
        "lazy_text_column": 0,
        "lazy_loading_start_line": 0,
        "dataloader_num_workers": 1
    }

def get_datasets(train_pth, test_pth, final_run=False):
    df_train = aa_as_pandas(get_aa_dataset(train_pth))
    df_test = aa_as_pandas(get_aa_dataset(test_pth))
    if final_run:
        # if this is the final run, use the evaluation set as part of the training set as well
        assert False, 'verify this functionality before continuing'
        df_val = aa_as_pandas(get_aa_dataset(train_pth.split('train')[0]+'val.csv'))
        df_train = pd.concat([df_train, df_val], ignore_index=True)

    return df_train, df_test


def get_model(model_path, num_labels, training_args, use_cuda=False, cuda_device=0, only_train_classifier=False):
    if only_train_classifier:
        # set the custom_parameter_groups arg properly, as well as the
        # train_custom_parameters_only=True
        logging.info(f'training only the classification layer')
        training_args.custom_parameter_groups = [{
            'params': ['classifier.weight', 'classifier.bias'],
            'lr': training_args["learning_rate"]
        }]
        training_args.train_custom_parameters_only = True
    logging.info(f'setting lr: {training_args["learning_rate"]}')
    model = ClassificationModel('bert' if 'tweet' not in model_path else 'bertweet',
                                model_path,
                                num_labels=num_labels,
                                args=training_args,
                                use_cuda=use_cuda,
                                cuda_device=cuda_device
                                )
    return model


def get_training_args(params=None):

    run_name = wandb.run.name

    params.output_dir = os.path.join(params.output_dir, run_name)
    params.best_model_dir = os.path.join(params.output_dir, run_name, 'best_model')

    training_args = {
        'reprocess_input_data': params.reprocess_input_data,  # reprocess the input data
        'num_train_epochs': params.num_train_epochs,  # number of epochs
        'evaluate_during_training': params.evaluate_during_training,  # run evaluation during training
        "use_early_stopping": params.use_early_stopping,
        "early_stopping_consider_epochs": params.early_stopping_consider_epochs,
        "early_stopping_delta": params.early_stopping_delta,
        "early_stopping_metric": params.early_stopping_metric,
        "early_stopping_metric_minimize": params.early_stopping_metric_minimize,
        "early_stopping_patience": params.early_stopping_patience,
        'evaluate_during_training_steps': params.evaluate_during_training_steps,  # steps in training before eval
        'evaluate_each_epoch': params.evaluate_each_epoch,
        'train_batch_size': params.train_batch_size,  # training batch size
        "eval_batch_size": params.eval_batch_size,  # evaluation batch size
        "gradient_accumulation_steps": params.gradient_accumulation_steps,  # steps before applying gradients
        "save_eval_checkpoints": params.save_eval_checkpoints,  # save evaluation checkpoints
        "save_steps": params.save_steps,
        "save_model_every_epoch": params.save_model_every_epoch,
        "save_best_model": params.save_best_model,
        "best_model_dir": params.best_model_dir,
        "output_dir": params.output_dir,
        "overwrite_output_dir": params.overwrite_output_dir,
        "learning_rate": params.lr,  # learning rate of our model
        "max_seq_length": params.max_seq_len,  # maximum sequence length in tokens
        "sliding_window": params.sliding_window,
        "stride": int(params.doc_stride * params.max_seq_len),  # stride when processing sentences
        "logging_steps": 1,  # the number of steps before logging
        "warmup_ratio": params.warmup_ratio,
        "warmup_steps": params.warmup_steps,
        "weight_decay": params.weight_decay,
        "manual_seed": params.seed,  # set the random seed
        "wandb_project": params.wandb_project,
        "lazy_loading": params.lazy_loading,
        "lazy_labels_column": 1,
        "lazy_text_column": 0,
        "lazy_loading_start_line": 0,
        "dataloader_num_workers": params.dataloader_num_workers
    }
    return training_args


def run_bertaa_sweep(config=None, train_pth=None, test_pth=None, device=None, project=None, use_cuda=True):
    # start the wandb sweep
    with wandb.init(config=config):
        config = wandb.config

        # get the datasets
        df_train, df_test = get_datasets(train_pth, test_pth)

        output_dir = 'BertAA_output'

        # get the model
        logging.info(f'setting lr: {config.learning_rate}')
        logging.info(f'setting output dir: {output_dir}')
        training_args = DEFAULT_TRAINING_ARGS
        training_args.update({
            'num_train_epochs': config.epochs,
            'reprocess_input_data': False,  # reprocess the input data
            'evaluate_during_training': True,  # run evaluation during training
            "use_early_stopping": True,
            "early_stopping_consider_epochs": False,
            "early_stopping_delta": 0,
            "early_stopping_metric": 'macro_accuracy',
            "early_stopping_metric_minimize": False,
            "early_stopping_patience": 5,
            'evaluate_during_training_steps': 2000,  # steps in training before eval
            'train_batch_size': config.batch_size,  # training batch size
            "gradient_accumulation_steps": 1,  # steps before applying gradients
            "save_eval_checkpoints": False,  # save evaluation checkpoints
            "save_steps": 2000,
            "save_model_every_epoch": False,
            "save_best_model": True,
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "eval_batch_size": config.batch_size,  # evaluation batch size
            "learning_rate": config.learning_rate,  # learning rate of our model
            "lr": config.learning_rate,  # just to make life easier
            "max_seq_length": config.max_seq_len,  # maximum sequence length in tokens
            "sliding_window": True,
            "stride": int(config.doc_stride * config.max_seq_len),  # stride when processing sentences
            "logging_steps": 1,  # the number of steps before logging
            "manual_seed": 0,  # set the random seed
            "seed": 0,  # again to make life easy
            "optimizer": config.optimizer,
            "warmup_ratio": 0,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "doc_stride": config.doc_stride,
            "max_seq_len": config.max_seq_len,
            "wandb_project": project,
        })

        best_model_dir = training_args['best_model_dir']
        tmp = training_args.copy()
        del tmp['output_dir']
        wandb.config.update(tmp)

        logging.info(f'building a model and running on GPU:{device}')
        model = get_model(model_path=config.model_path, num_labels=df_train['labels'].nunique(),
                          training_args=training_args, use_cuda=use_cuda, cuda_device=device,
                          only_train_classifier=False)

        # now train the model for the number of epochs we care about
        model.train_model(df_train,
                          args=training_args,
                          eval_df=df_test,
                          accuracy=metrics.accuracy_score,
                          macro_accuracy=metrics.balanced_accuracy_score,
                          micro_recall=partial(metrics.recall_score, average='micro'),
                          macro_recall=partial(metrics.recall_score, average='macro'),
                          micro_precision=partial(metrics.precision_score, average='micro'),
                          macro_precision=partial(metrics.precision_score, average='macro'),
                          micro_f1=partial(metrics.f1_score, average='micro'),
                          macro_f1=partial(metrics.f1_score, average='macro')
                          )

        wandb.save(best_model_dir)


def lim_dset(dset_list, auth_lim, sample_lim):
    selected_data = dset_list
    # limit the number of authors if needed (sorted by auth num for reproducibility)
    if auth_lim is not None:
        selected_data = [[a, t] for i, (a, t) in enumerate(dset_list) if i < auth_lim]

    # limit the number of samples per author
    if sample_lim is not None:
        selected_dict = list_dset_to_dict(selected_data)
        for k in list(selected_dict.keys()):
            selected_dict[k] = selected_dict[k][:sample_lim]
        selected_data = dict_dset_to_list(selected_dict)
    return selected_data

def run_bertaa(params):

    wandb.login()
    wandb_project = params.wandb_project
    params.model = 'BERTweet' if 'tweet' in params.model_path.lower() else 'BertAA'
    wandb.init(project=wandb_project, config=vars(params))

    if params.lazy_loading:
        assert params.author_limit is None, 'Limiting the number of authors is not supported with lazy loading'
        assert params.sample_per_author_limit is None, 'Limiting the number of authors is not supported with lazy loading'
        logging.info('using lazy loading')
        df_train = params.train_dataset
        df_eval = params.eval_dataset
        df_test = params.test_dataset
        num_labels = params.num_labels
    else:
        df_train = lim_dset(get_aa_dataset(params.train_dataset), params.author_limit, params.sample_per_author_limit)
        df_train = aa_as_pandas(df_train)
        df_eval = lim_dset(get_aa_dataset(params.eval_dataset), params.author_limit, params.sample_per_author_limit)
        df_eval = aa_as_pandas(df_eval)
        df_test = lim_dset(get_aa_dataset(params.test_dataset), params.author_limit, params.sample_per_author_limit)
        df_test = aa_as_pandas(df_test)


        print(np.min(df_train['labels'].unique()))
        print(np.max(df_train['labels'].unique()))
        print('number unique training labels:')
        print(df_train['labels'].nunique())
        print('-------------')
        print(np.min(df_eval['labels'].unique()))
        print(np.max(df_eval['labels'].unique()))
        print('number unique eval labels:')
        print(df_eval['labels'].nunique())
        print('-------------')
        print(np.min(df_test['labels'].unique()))
        print(np.max(df_test['labels'].unique()))
        print('number unique testing labels:')
        print(df_test['labels'].nunique())
        print('-------------')
        train_labels = df_train['labels'].unique()
        val_labels = df_eval['labels'].unique()
        test_labels = df_test['labels'].unique()

        for lbl in val_labels:
            if lbl not in train_labels:
                print(f'!!!!!{lbl} from val set not in train set. . . ')

        for lbl in test_labels:
            if lbl not in train_labels:
                print(f'!!!!!{lbl} from test set not in train set. . . ')

        if params.final_run:
            df_train = pd.concat([df_train, df_eval], ignore_index=True)

        num_labels = df_train['labels'].nunique()

    training_args = get_training_args(params)
    tmp = training_args.copy()
    del tmp['output_dir']
    wandb.config.update(tmp)

    model = get_model(model_path=args.model_path, num_labels=num_labels, training_args=training_args,
                      use_cuda=not params.no_cuda, cuda_device=params.device)

    print(model.model)

    if 'pan20' in params.test_dataset:
        logging.info('treating this as a special case for metrics. . . ')
        model.train_model(df_train,
                          eval_df=df_eval if not params.final_run else df_test,
                          accuracy=metrics.accuracy_score,
                          macro_accuracy=metrics.balanced_accuracy_score
                          )
    else:
        model.train_model(df_train,
                          eval_df=df_eval if not params.final_run else df_test,
                          accuracy=metrics.accuracy_score,
                          macro_accuracy=metrics.balanced_accuracy_score,
                          micro_recall=partial(metrics.recall_score, average='micro'),
                          macro_recall=partial(metrics.recall_score, average='macro'),
                          micro_precision=partial(metrics.precision_score, average='micro'),
                          macro_precision=partial(metrics.precision_score, average='macro'),
                          micro_f1=partial(metrics.f1_score, average='micro'),
                          macro_f1=partial(metrics.f1_score, average='macro')
                          )

    if not params.final_run:
        if 'pan20' in params.test_dataset:
            model.eval_model(df_eval,
                             accuracy=metrics.accuracy_score,
                             macro_accuracy=metrics.balanced_accuracy_score,
                             )
        else:
            model.eval_model(df_eval,
                             accuracy=metrics.accuracy_score,
                             macro_accuracy=metrics.balanced_accuracy_score,
                             micro_recall=partial(metrics.recall_score, average='micro'),
                             macro_recall=partial(metrics.recall_score, average='macro'),
                             micro_precision=partial(metrics.precision_score, average='micro'),
                             macro_precision=partial(metrics.precision_score, average='macro'),
                             micro_f1=partial(metrics.f1_score, average='micro'),
                             macro_f1=partial(metrics.f1_score, average='macro')
                             )

    predictions, raw_outputs = model.predict(df_test['text'].tolist())
    if 'pan20' in params.test_dataset:
        test_results = eval_metrics.aa_metrics(df_test['labels'], predictions, raw_outputs, prefix='test/',
                                               no_auc=True, special=True)
    else:
        test_results = eval_metrics.aa_metrics(df_test['labels'], predictions, raw_outputs, prefix='test/', no_auc=True)

    wandb.log(test_results)

    # save the best model
    # TODO: add path to best_model save path
    wandb.save(f'{params.best_model_dir}-final')


if __name__ == "__main__":

    # get command line args
    parser = argparse.ArgumentParser(description='Run a BertAA model from the command line')

    parser.add_argument('--wandb_project', type=str, default='imdb62',
                        help='the mlflow experiment name')
    parser.add_argument('--train_dataset', type=str,
                        default='/home/jtyo/data/Projects/On_the_SOTA_of_Authorship_Verification/datasets/imdb'
                                '/processed/imdb62/imdb62_train.csv')
    parser.add_argument('--test_dataset', type=str,
                        default='/home/jtyo/data/Projects/On_the_SOTA_of_Authorship_Verification/datasets/imdb'
                                '/processed/imdb62/imdb62_AA_test.csv')
    parser.add_argument('--eval_dataset', type=str,
                        default='/home/jtyo/data/Projects/On_the_SOTA_of_Authorship_Verification/datasets/imdb'
                                '/processed/imdb62/imdb62_AA_val.csv')
    parser.add_argument('--model_path', type=str, default='bert-base-cased')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--evaluate_during_training', action='store_true')
    parser.add_argument('--evaluate_during_training_steps', type=int, default=-1)
    parser.add_argument('--evaluate_each_epoch', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.15)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--reprocess_input_data', action='store_true')
    parser.add_argument('--save_eval_checkpoints', action='store_true')
    parser.add_argument('--save_model_every_epoch', action='store_true')
    parser.add_argument('--save_steps', type=int, default=-1)
    parser.add_argument('--save_best_model', action='store_true')
    parser.add_argument('--doc_stride', type=float, default=0.8, help='express as % of max_seq_len')
    parser.add_argument('--sliding_window', action='store_true', help='break long samples into multiple samples')
    parser.add_argument('--output_dir', type=str, default='BertAA_outputs/')
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--use_early_stopping', action='store_true')
    parser.add_argument('--early_stopping_consider_epochs', action='store_true')
    parser.add_argument('--early_stopping_delta', type=float, default=0.01)
    parser.add_argument('--early_stopping_metric', type=str, default='macro_accuracy')
    parser.add_argument('--early_stopping_metric_minimize', action='store_true')
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--final_run', action='store_true', help='use eval set as part of training')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--lazy_loading', action='store_true')
    parser.add_argument('--num_labels', type=int, default=None)
    parser.add_argument('--only_train_classifier', action='store_true')
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--author_limit', type=int, default=None)
    parser.add_argument('--sample_per_author_limit', type=int, default=None)
    parser.set_defaults(final_run=False, early_stopping_metric_minimize=False, early_stopping_consider_epochs=False,
                        use_early_stopping=False, overwrite_output_dir=False, save_best_model=False,
                        save_model_every_epoch=False, save_eval_checkpoints=False, reprocess_input_data=False,
                        evaluate_during_training=False, sliding_window=False, no_cuda=False,
                        only_train_classifier=False, lazy_loading=False)

    args = parser.parse_args()

    # sloppy but makes the sweep arg configuration easier
    args.num_train_epochs = args.epochs

    run_bertaa(args)
