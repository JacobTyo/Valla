from sklearn import metrics
from valla.utils import eval_metrics
import wandb
import os
import pandas as pd
import numpy as np
from functools import partial
from valla.dsets.loaders import aa_as_pandas, get_aa_dataset
from valla.methods.meta_methods.MetaClassificationModel import MetaClassificationModel
import argparse
import logging
import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (6144, rlimit[1]))

logging.basicConfig(level=logging.DEBUG)


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
    model = MetaClassificationModel('bert',
                                    model_path,
                                    num_labels=num_labels,
                                    args=training_args,
                                    use_cuda=use_cuda,
                                    cuda_device=cuda_device
                                    )
    # print(model.model)
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
        "dataloader_num_workers": params.dataloader_num_workers,
        "ways": params.ways,
        "shots": params.shots,
        "inner_lr": params.inner_lr,
        "meta_batch_size": params.meta_batch_size,
        "num_outer_steps": params.num_outer_steps,
        "num_inner_updates": params.num_inner_updates,
        "use_multiprocessing": params.use_multiprocessing,
        "use_multiprocessing_for_evaluation": False,
        "meta_type": params.meta_type
    }
    return training_args


def run_meta_bertaa(params):

    wandb.login()
    wandb_project = params.wandb_project
    params.model = 'MetaBertAA'
    wandb.init(project=wandb_project, config=vars(params))

    df_train = aa_as_pandas(get_aa_dataset(params.train_dataset))
    meta_test_train_df = aa_as_pandas(get_aa_dataset(params.meta_test_train_dset))
    meta_test_test_df = aa_as_pandas(get_aa_dataset(params.meta_test_test_dset))

    # if params.final_run:
    #     df_train = pd.concat([df_train, df_eval], ignore_index=True)

    num_labels = meta_test_train_df['labels'].nunique()

    logging.info(f'there are {num_labels} labels')

    training_args = get_training_args(params)
    tmp = training_args.copy()
    del tmp['output_dir']
    wandb.config.update(tmp)

    model = get_model(model_path=args.model_path, num_labels=num_labels, training_args=training_args,
                      use_cuda=not params.no_cuda, cuda_device=params.device)

    # print(model.model)

    model.meta_train_model(df_train,
                           meta_test_train_df=meta_test_train_df,
                           meta_test_test_df=meta_test_test_df,
                           accuracy=metrics.accuracy_score,
                           macro_accuracy=metrics.balanced_accuracy_score,
                           micro_recall=partial(metrics.recall_score, average='micro'),
                           macro_recall=partial(metrics.recall_score, average='macro'),
                           micro_precision=partial(metrics.precision_score, average='micro'),
                           macro_precision=partial(metrics.precision_score, average='macro'),
                           micro_f1=partial(metrics.f1_score, average='micro'),
                           macro_f1=partial(metrics.f1_score, average='macro')
                           )


        # model.eval_model(df_eval,
        #                  accuracy=metrics.accuracy_score,
        #                  macro_accuracy=metrics.balanced_accuracy_score,
        #                  micro_recall=partial(metrics.recall_score, average='micro'),
        #                  macro_recall=partial(metrics.recall_score, average='macro'),
        #                  micro_precision=partial(metrics.precision_score, average='micro'),
        #                  macro_precision=partial(metrics.precision_score, average='macro'),
        #                  micro_f1=partial(metrics.f1_score, average='micro'),
        #                  macro_f1=partial(metrics.f1_score, average='macro')
        #                  )

    # predictions, raw_outputs = model.predict(df_test['text'].tolist())
    # test_results = eval_metrics.aa_metrics(df_test['labels'], predictions, raw_outputs, prefix='test/', no_auc=True)

    # wandb.log(test_results)

    # save the best model
    # TODO: add path to best_model save path
    # wandb.save(f'{params.best_model_dir}-final')


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
    parser.add_argument('--output_dir', type=str, default='MetaBertAA_outputs/')
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
    parser.add_argument('--ways', type=int, default=5)
    parser.add_argument('--shots', type=int, default=1)
    parser.add_argument('--inner_lr', type=float, default=1e-2, help='meta learning rate')
    parser.add_argument('--meta_batch_size', type=int, default=5, help='the number of tasks per outer update')
    parser.add_argument('--num_outer_steps', type=int, default=1000, help='the number of outer updates per epoch '
                                                                          '(tasks sampled randomly)')
    parser.add_argument('--num_inner_updates', type=int, default=1, help='the number of update steps for the adapt '
                                                                         'phase on each meta batch')
    parser.add_argument('--meta_test_train_dset', type=str, default=None)
    parser.add_argument('--meta_test_test_dset', type=str, default=None)
    parser.add_argument('--meta_type', type=str, default='ANIL')
    parser.add_argument('--use_multiprocessing', action='store_true')
    parser.set_defaults(final_run=False, early_stopping_metric_minimize=False, early_stopping_consider_epochs=False,
                        use_early_stopping=False, overwrite_output_dir=False, save_best_model=False,
                        save_model_every_epoch=False, save_eval_checkpoints=False, reprocess_input_data=False,
                        evaluate_during_training=False, sliding_window=False, no_cuda=False,
                        only_train_classifier=False, lazy_loading=False, use_multiprocessing=False)

    args = parser.parse_args()

    # sloppy but makes the sweep arg configuration easier
    args.num_train_epochs = args.epochs

    run_meta_bertaa(args)
