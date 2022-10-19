"""
This file if a modified version of the one found here:
    https://github.com/pan-webis-de/muttenthaler19/blob/master/AuthorshipAttribution.ipynb
"""
import os
import pickle
import re
import json
import argparse
import time
import logging
import numpy as np
import wandb
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
# from sklearn.svm import SVC  # used in the original implementation but very slow on large datasets
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from valla.dsets import loaders
from valla.utils import eval_metrics
from typing import List, Callable, Tuple

logging.basicConfig(level=logging.DEBUG)


def base_preprocessor(string: str) -> str:
    """
    Function that computes regular expressions.
    """
    string = re.sub("[0-9]", "0", string)  # each digit will be represented as a 0
    string = re.sub(r'( \n| \t)+', '', string)
    # text = re.sub("[0-9]+(([.,^])[0-9]+)?", "#", text)
    string = re.sub("https:\\\+([a-zA-Z0-9.]+)?", "@", string)
    return string


def char_diff_preprocessor(string: str) -> str:
    """
    Function that computes regular expressions.
    """
    string = base_preprocessor(string)
    string = re.sub("[a-zA-Z]+", "*", string)
    # string = ''.join(['*' if char.isalpha() else char for char in string])
    return string


def word_preprocessor(string: str) -> str:
    """
    Function that computes regular expressions.
    """
    string = base_preprocessor(string)
    # if model is a word n-gram model, remove all punctuation
    string = ''.join([char for char in string if char.isalnum() or char.isspace()])
    return string


def get_vectorizers(analyzer: str = 'char',
                    gram_range: List = (1, 2),
                    preprocessor: Callable = base_preprocessor,
                    max_features: int = 1000,
                    min_df: float = 0.1,
                    smooth_idf: bool = True,
                    sublinear_tf: bool = True) -> Tuple[CountVectorizer, TfidfTransformer]:
    """
    Get a vectorizer for this project
    """
    logging.debug(f'Building a {gram_range} TfidfVectorizer for {analyzer} with the {preprocessor} preprocessor.')
    logging.debug(f'Other params:\n\t\tmax_features: {max_features}\n\t\tmin_df: {min_df}\n\t\tsmooth_idf: '
                  f'{smooth_idf}\n\t\tsublinear_tf: {sublinear_tf}')
    count_vectorizer = CountVectorizer(decode_error='ignore', strip_accents='unicode', lowercase=False, stop_words=None,
                                       ngram_range=gram_range, analyzer=analyzer, min_df=min_df,
                                       max_features=max_features)
    tfidf_vectorizer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
    return count_vectorizer, tfidf_vectorizer


def ngram(analyzer: str, train_texts: List, train_labels: List, test_texts: List, test_labels: List, gram_range: List,
          preprocessor: Callable, max_features: int, min_df: float, sublinear_tf: bool, use_lsa: bool, lsa_factors: int,
          dual: bool, log_prefix: str, save_path: str = None, project: str = '', logistic_regression: bool = False,
          num_workers: int = 1):
    logging.info(f'{analyzer}: building the tf-idf vectorizer for the {analyzer} n-gram model')
    count_vectorizer, tfidf_transformer = get_vectorizers(analyzer=analyzer if 'dist' not in analyzer else 'char',
                                                          gram_range=gram_range,
                                                          preprocessor=preprocessor,
                                                          max_features=max_features,
                                                          min_df=min_df,
                                                          smooth_idf=True,
                                                          sublinear_tf=sublinear_tf)

    # cache the vectorizer, just load it if the params match up
    count_vectorizer_cache_path = f'ngram_cache/cv_{project}_{analyzer}_{gram_range[0]}-{gram_range[1]}_' \
                                  f'{max_features}_{min_df}.pkl'
    tfidf_vectorizer_cache_path = f'ngram_cache/idf_{project}_{analyzer}_{gram_range[0]}-{gram_range[1]}_' \
                                  f'{max_features}_{min_df}_{sublinear_tf}.pkl'

    # fit the count vectorizer
    if os.path.isfile(count_vectorizer_cache_path):
        logging.info(f'loading the pre-fit count vectorizer from {count_vectorizer_cache_path}')
        start = time.time()
        with open(count_vectorizer_cache_path, 'rb') as f:
            count_vectorizer = pickle.load(f)
        logging.debug(f'took {(time.time() - start) / 60} minutes')
        logging.info(f'transforming the texts with the pre-fit vectorizer.')
        train_term_matrix = count_vectorizer.transform(train_texts)

    else:
        logging.info(f'{analyzer}: fitting the count vectorizer')
        start = time.time()
        train_term_matrix = count_vectorizer.fit_transform(train_texts).toarray()
        logging.info(f'saving count vectorizer to cache: {count_vectorizer_cache_path}')
        os.makedirs(os.path.dirname(count_vectorizer_cache_path), exist_ok=True)
        with open(count_vectorizer_cache_path, 'wb') as f:
            pickle.dump(count_vectorizer, f)
        logging.debug(f'took {(time.time() - start) / 60} minutes')

    # fit the tfidf transformer
    if os.path.isfile(tfidf_vectorizer_cache_path):
        logging.info(f'loading the pre-fit tfidf vectorizer from {tfidf_vectorizer_cache_path}')
        start = time.time()
        with open(tfidf_vectorizer_cache_path, 'rb') as f:
            tfidf_transformer = pickle.load(f)
        logging.debug(f'took {(time.time() - start) / 60} minutes')
        logging.info(f'transforming the training texts with the  tfidf transformer')
        train_data = tfidf_transformer.transform(train_term_matrix)
    else:
        logging.info(f'{analyzer}: fitting the tfidf vectorizer')
        start = time.time()
        train_data = tfidf_transformer.fit_transform(train_term_matrix).toarray()
        logging.info(f'saving tfidf vectorizer to cache: {tfidf_vectorizer_cache_path}')
        os.makedirs(os.path.dirname(tfidf_vectorizer_cache_path), exist_ok=True)
        with open(tfidf_vectorizer_cache_path, 'wb') as f:
            pickle.dump(tfidf_transformer, f)
        logging.debug(f'took {(time.time() - start) / 60} minutes')


    logging.info(f'{analyzer}: vectorizing the test texts')
    test_data = tfidf_transformer.transform(count_vectorizer.transform(test_texts).toarray()).toarray()

    logging.info(f'{analyzer}: scaling the vectorized data')
    max_abs_scaler = preprocessing.MaxAbsScaler()
    scaled_train_data = max_abs_scaler.fit_transform(train_data)
    scaled_test_data = max_abs_scaler.transform(test_data)

    if use_lsa:
        lsa_cache_path = f'ngram_cache/lsa_{project}_{analyzer}_{gram_range[0]}-{gram_range[1]}_{max_features}_' \
                                f'{min_df}_{sublinear_tf}_{lsa_factors}.pkl'
        if os.path.isfile(lsa_cache_path):
            logging.info(f'loading the svd transform from cache')
            start = time.time()
            with open(lsa_cache_path, 'rb') as f:
                svd = pickle.load(f)
            scaled_train_data = svd.transform(scaled_train_data)
            scaled_test_data = svd.transform(scaled_test_data)
            logging.debug(f'took {(time.time() - start) / 60} minutes')
        else:
            logging.info(f'{analyzer}: reducing demensionality with TruncatedSVD')
            start = time.time()
            svd = TruncatedSVD(n_components=lsa_factors, algorithm='randomized', random_state=0)
            # Char
            scaled_train_data = svd.fit_transform(scaled_train_data)
            scaled_test_data = svd.transform(scaled_test_data)
            logging.debug(f'took {(time.time() - start) / 60} minutes')
            # cache the svd
            with open(lsa_cache_path, 'wb') as f:
                pickle.dump(svd, f)

    logging.info(f'{analyzer}: fitting the classifier')
    start = time.time()
    # This was the classifier used in the original implementation, but we need a more efficient one
    # char_std = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear',
    #                                                           gamma='auto', verbose=True)))
    if logistic_regression:
        # classifier = LogisticRegression(multi_class='multinomial', dual=dual)
        classifier = SGDClassifier(loss='log', n_jobs=num_workers, early_stopping=False, verbose=1)
    else:
        classifier = LogisticRegression(multi_class='multinomial', dual=dual)

    classifier.fit(scaled_train_data, train_labels)
    logging.debug(f'took {(time.time() - start) / 60} minutes')

    logging.info(f'{analyzer}: inference on the test set')
    start - time.time()
    predictions = classifier.predict(scaled_test_data)
    predicted_probs = classifier.predict_proba(scaled_test_data)
    logging.debug(f'took {(time.time() - start) / 60} minutes')

    # compute and log char ngram
    logging.info(f'{analyzer}: logging to wandb')
    wandb.sklearn.plot_classifier(classifier,
                                  scaled_train_data, scaled_test_data,
                                  train_labels, test_labels,
                                  predictions, predicted_probs,
                                  [x for x in range(len(set(train_labels)))],
                                  is_binary=False,
                                  model_name=analyzer)
    results = eval_metrics.aa_metrics(test_labels, predictions, predicted_probs, prefix=log_prefix, no_auc=True)
    wandb.log(results)

    # save the model
    clf_name = 'logreg_sgd' if logistic_regression else 'logreg'
    svm_path = os.path.join(os.path.dirname(tfidf_vectorizer_cache_path), f'{analyzer}_{clf_name}.pkl')
    logging.debug(f'saving the {analyzer}_{clf_name} to {svm_path}')
    with open(svm_path, 'wb') as f:
        pickle.dump(classifier, f)

    wandb.save(svm_path)

    return predicted_probs


def run_ngram(config={}, ngram_type: str = 'char', train_pth: str = None, val_pth: str = None, test_pth: str = None,
              project='', num_workers=10):

    # need to make sure config is a namespace
    if isinstance(config, dict):
        config = argparse.Namespace(**config)

    sweep = True if project != '' else False
    project = project if project != '' else config.project

    tmp = vars(config)
    tmp['model'] = ngram_type

    with wandb.init(project=project, config=tmp, reinit=True):
        if sweep:
            config = wandb.config
            config.project = project
            config.num_workers = num_workers

        # config.project = project
        # config.save_path = os.path.join('ngram', project, wandb.run.name)

        # config.model = ngram_type
        # wandb.config.update(config)

        # add the run name to make sure we don't overwrite other models
        # if config.save_path is not None:
        save_path = os.path.join('ngram', project, wandb.run.name)

        logging.info('starting')

        # get the training and testing dataset as List[List[Union[int, str]]]
        logging.info('loading the datasets')
        train_dset = loaders.get_aa_dataset(train_pth)
        test_dset = loaders.get_aa_dataset(test_pth)

        if val_pth is not None and val_pth != '':
            log_prefix = 'test/'
            train_dset.extend(loaders.get_aa_dataset(val_pth))
        else:
            log_prefix = 'val/'

        train_texts = [text for _, text in train_dset]
        train_labels = [label for label, _ in train_dset]
        test_texts = [text for _, text in test_dset]
        test_labels = [label for label, _ in test_dset]

        # get the proper preprocessor and make sure ngram_type is set for the vectorizer
        if sweep:
            gram_range = config.gram_range
        if ngram_type == 'char':
            preprocessor = base_preprocessor
            if not sweep:
                gram_range = config.char_range
        elif ngram_type == 'dist_char':
            preprocessor = char_diff_preprocessor
            if not sweep:
                gram_range = config.dist_range
        elif ngram_type == 'word':
            preprocessor = word_preprocessor
            if not sweep:
                gram_range = config.word_range
        else:
            raise ValueError(f'ngram_type was not set properly, should be in [char, dist_char, word], got {ngram_type}')

        probas = ngram(analyzer=ngram_type, train_texts=train_texts, train_labels=train_labels, test_texts=test_texts,
                       test_labels=test_labels, gram_range=gram_range, preprocessor=preprocessor,
                       max_features=config.max_features,
                       min_df=config.min_df, sublinear_tf=config.sublinear_tf, use_lsa=config.use_lsa,
                       lsa_factors=config.lsa_factors, dual=not config.primal, log_prefix=log_prefix,
                       save_path=save_path, project=config.project, logistic_regression=config.logistic_regression,
                       num_workers=config.num_workers)

    return probas


def ensemble(config, test_labels, probas_word, probas_char, probas_dist, prefix=''):
    # need to make sure config is a namespace
    if isinstance(config, dict):
        config = argparse.Namespace(**config)

    config.model = 'ensemble'

    # compute and long ensemble
    with wandb.init(project=args.project, config=vars(config), reinit=True):

        logging.info('ensembling the models')
        # Soft Voting procedure (combines the votes of the three individual classifier)
        avg_probas = np.average([probas_word, probas_dist, probas_char], axis=0)
        avg_predictions = []
        for text_probs in avg_probas:
            ind_best = np.argmax(text_probs)
            avg_predictions.append(ind_best)

        ensemble_results = eval_metrics.aa_metrics(test_labels, avg_predictions, avg_probas, prefix=prefix, no_auc=True)
        wandb.log(ensemble_results)
        wandb.finish()
        logging.info('done')


if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser(description='Run a N-Gram model from the command line')

    parser.add_argument('--project', type=str, default='imdb62',
                        help='the mlflow experiment name')
    parser.add_argument('--train_dataset', type=str,
                        default='/home/jtyo/data/Projects/On_the_SOTA_of_Authorship_Verification/datasets/imdb'
                                '/processed/imdb62/imdb62_train.csv')
    parser.add_argument('--val_dataset', type=str, default='')
    parser.add_argument('--test_dataset', type=str,
                        default='/home/jtyo/data/Projects/On_the_SOTA_of_Authorship_Verification/datasets/imdb'
                                '/processed/imdb62/imdb62_AA_test.csv')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--seed', metavar='seed', type=int, default=0)
    parser.add_argument('--word_range', nargs='+', type=int, default=[1, 3])
    parser.add_argument('--dist_range', nargs='+', type=int, default=[1, 3])
    parser.add_argument('--char_range', nargs='+', type=int, default=[2, 5])
    parser.add_argument('--n_best_factor', type=float, default=0.5)
    parser.add_argument('--pt', type=float, default=0.1)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--use_lsa', action='store_true')
    parser.add_argument('--lsa_factors', type=int, default=63)
    parser.add_argument('--sublinear_tf', action='store_true')
    parser.add_argument('--primal', action='store_true')
    parser.add_argument('--max_features', type=int, default=100_000)
    parser.add_argument('--min_df', type=float, default=0.01)
    parser.add_argument('--type', type=str, default='')
    parser.add_argument('--logistic_regression', action='store_true')
    parser.add_argument('--num_workers', type=int, default=10)

    args = parser.parse_args()

    args.word_range = tuple(args.word_range)
    args.dist_range = tuple(args.dist_range)
    args.char_range = tuple(args.char_range)

    np.random.seed(args.seed)

    wandb.login()

    log_prf = 'test' if 'test' in args.test_dataset else 'val'

    total_time_start = time.time()

    if args.type == '':
        char_probas = run_ngram(args, 'char', train_pth=args.train_dataset, val_pth=args.val_dataset,
                                test_pth=args.test_dataset)
        dist_probas = run_ngram(args, 'dist_char', train_pth=args.train_dataset, val_pth=args.val_dataset,
                                test_pth=args.test_dataset)
        word_probas = run_ngram(args, 'word', train_pth=args.train_dataset, val_pth=args.val_dataset,
                                test_pth=args.test_dataset)
        # now ensemble the results
        test_lbls = [lbl for lbl, _ in loaders.get_aa_dataset(args.test_dataset)]
        ensemble(args, test_labels=test_lbls, probas_char=char_probas, probas_dist=dist_probas, probas_word=word_probas)
    else:
        run_ngram(args, args.type, train_pth=args.train_dataset, val_pth=args.val_dataset, test_pth=args.test_dataset)

    logging.info(f'this run took {(time.time() - total_time_start)/60} minutes')
