import random
import csv
import os
import numpy as np
import json

import pandas as pd
from tqdm import tqdm
from typing import List, Union, Dict
import logging
from valla.dsets.loaders import av_as_pandas, aa_as_pandas

logging.basicConfig(level=logging.INFO)


def list_dset_to_dict(data: List[List[Union[int, str]]]) -> Dict:
    out = {}
    for auth, text in data:
        out.setdefault(auth, []).append(text)
    return out


def dict_dset_to_list(data: Dict) -> List[List[Union[int, str]]]:
    out = []
    for auth, texts in data.items():
        for text in texts:
            out.append([auth, text])
    return out


def auth_text_make_unique(data: Dict):
    unique = {}
    for author, texts in data.items():
        unique[author] = list(set(texts))
    return unique


def sample_av_dataset(data: Dict, txt_size_lim=None) -> List[List[Union[int, str, str]]]:
    sampled_dset = []
    for auth, texts in tqdm(data.items()):
        for text in texts:
            # get a same_author sample - if not enough texts just skip
            if len(texts) < 2:
                continue
            same_auth_txt = random.choice(data[auth])
            while text == same_auth_txt:
                same_auth_txt = random.choice(data[auth])
            if txt_size_lim is None:
                sampled_dset.append([1, text, same_auth_txt])
            else:
                sampled_dset.append([1, text[:txt_size_lim], same_auth_txt[:txt_size_lim]])

            # get a different_author sample
            diff_auth = random.choice(list(data.keys()))
            while auth == diff_auth:
                diff_auth = random.choice(list(data.keys()))
            diff_auth_txt = random.choice(data[diff_auth])
            if txt_size_lim is None:
                sampled_dset.append([0, text, diff_auth_txt])
            else:
                sampled_dset.append([0, text[:txt_size_lim], same_auth_txt[:txt_size_lim]])

    return sampled_dset


def write_aa_dataset(data: Dict, file_path: str) -> None:
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['author_id', 'text'])
        for auth, texts in data.items():
            for text in texts:
                writer.writerow([auth, text])


def write_av_dataset(data: List[List[Union[int, str, str]]], file_path: str) -> None:
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['same/diff auth', 'text1', 'text2'])
        writer.writerows(data)


def lim_av_size(data):
    logging.debug('liming the size of each sample to 100,000 chars')
    was_a_df = True
    if not isinstance(data, pd.DataFrame):
        data = av_as_pandas(data)
        was_a_df = False

    data['text0'] = data['text0'].str.slice(stop=100_000)
    data['text1'] = data['text1'].str.slice(stop=100_000)

    if was_a_df:
        return data
    else:
        return data.values.tolist()


def lim_aa_size(data):
    logging.debug('liming the size of each sample to 100,000 chars')
    was_a_df = True
    if not isinstance(data, pd.DataFrame):
        data = aa_as_pandas(data)
        was_a_df = False

    data['text'] = data['text'].str.slice(stop=100_000)

    if was_a_df:
        return data
    else:
        return data.values.tolist()


def summarize_dataset(original_data: Dict,
                      train_data: Dict,
                      aa_val_data: Dict,
                      aa_test_data: Dict,
                      av_val_data: List[List[Union[int, str, str]]],
                      av_test_data: List[List[Union[int, str, str]]],
                      save_file: str) -> None:
    # things we care about:
    #  - number of authors
    #  - number of documents
    #  - number of training documents
    #  - number of AA validation documents
    #  - number of AA testing documents
    #  - number of AV validation pairs
    #  - number of AV testing pairs
    #  - number of documents per author (max, min, median, mean, mode, std)
    #  - number of words per document (max, min, median, mdean, mode, std)
    #  - number of words per author (max, min, median, mdean, mode, std)
    #  - The previous 3 things repeated for each set

    stats = {}

    names = ['all', 'train', 'aa_val', 'aa_test']

    for d, set_lbl in zip([original_data, train_data, aa_val_data, aa_test_data], names):
        if d:
            authors = list(d.keys())
            num_authors = len(authors)
            doc_lens_per_auth = [len(texts) for texts in d.values()]
            num_docs = sum(doc_lens_per_auth)

            words_per_auth = []
            for texts in d.values():
                words = 0
                for text in texts:
                    words += len(text.split())
                words_per_auth.append(words)

            num_words = sum(words_per_auth)

            stats[set_lbl + ' number of authors'] = num_authors
            stats[set_lbl + ' number of documents'] = num_docs
            stats[set_lbl + ' number of words'] = num_words
            stats[set_lbl + ' docs per author'] = num_docs / num_authors
            stats[set_lbl + ' max docs per author'] = int(np.max(doc_lens_per_auth))
            stats[set_lbl + ' min docs per author'] = int(np.min(doc_lens_per_auth))
            stats[set_lbl + ' mean docs per author'] = float(np.mean(doc_lens_per_auth))
            stats[set_lbl + ' median docs per author'] = int(np.median(doc_lens_per_auth))
            stats[set_lbl + ' std docs per author'] = float(np.std(doc_lens_per_auth))
            stats[set_lbl + ' words per author'] = num_words / num_authors
            stats[set_lbl + ' max words per author'] = int(np.max(words_per_auth))
            stats[set_lbl + ' min words per author'] = int(np.min(words_per_auth))
            stats[set_lbl + ' mean words per author'] = float(np.mean(words_per_auth))
            stats[set_lbl + ' median words per author'] = float(np.median(words_per_auth))
            stats[set_lbl + ' std words per author'] = float(np.std(words_per_auth))

    # now for AV we care about:
    #   number positive pairs
    #   number negative pairs
    av_names = ['av_val', 'av_test']

    for d, d_lbl in zip([av_val_data, av_test_data], av_names):
        if d:
            lbl_list = [int(lbl) for lbl, _, _ in d]
            pos_pairs = sum(lbl_list)
            neg_pairs = sum((-1*np.asarray(lbl_list)+1).tolist())
            stats[d_lbl + ' number positive pairs'] = pos_pairs
            stats[d_lbl + ' number negative pairs'] = neg_pairs

    with open(save_file, 'w+') as f:
        json.dump(stats, f, indent=4, sort_keys=True)


def save_aa_and_av_dset(names, data, save_path, dataset_name):
    av_dsets = [None, None]
    for i, (name, dset) in enumerate(zip(names, data)):
        if dset:
            aa_fp = os.path.join(save_path, f'{dataset_name}_AA_{name}.csv')
            logging.info(f'writing the AA {name} set to {aa_fp}')
            write_aa_dataset(dset, aa_fp)

            logging.info(f'sampling the AV {name} set')
            av = sample_av_dataset(dset)
            av_dsets[i] = av
            av_fp = os.path.join(save_path, f'{dataset_name}_AV_{name}.csv')
            logging.info(f'writing the AV {name} set to {aa_fp}')
            write_av_dataset(av, av_fp)
    return av_dsets


def finalize_dataset(original_data: Dict, train: Dict, val: Dict, test: Dict, dataset_name, save_path):

    # make sure save path exists and is empty
    if os.path.exists(save_path) and len(os.listdir(save_path)) > 1:
        assert False, f'{save_path} is not an empty direction, delete and rerun to continue.'
    # create the dir if not
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # treat separately for error handling of new dataset configs
    train_fp = os.path.join(save_path, f'{dataset_name}_train.csv')
    logging.info(f'writing the training set to {train_fp}')
    write_aa_dataset(train, train_fp)

    # Sample the eval set if not none
    av_dsets = save_aa_and_av_dset(['test', 'val'], [test, val], save_path, dataset_name)

    # also take advantage of this opportunity to do some dataset analysis and write to file.
    stats_path = os.path.join(save_path, f'{dataset_name}_stats.json')
    logging.info(f'calculating data statistics and writing to {stats_path}')
    summarize_dataset(original_data, train, val, test, av_dsets[0], av_dsets[1], stats_path)

    logging.info('complete')


def finalize_cross_dataset(original_data: Dict, train: Dict, val: Dict, test: Dict, cross_topic_val: Dict,
                           cross_topic_test: Dict, cross_genre_val: Dict = None, cross_genre_test: Dict = None,
                           dataset_name='', save_path=''):

    finalize_dataset(original_data, train, val, test, dataset_name, save_path)

    logging.info('logging cross data')
    # now also deal with the cross_topic and cross_genre datasets
    cross_topic_av = save_aa_and_av_dset(['cross_topic_val', 'cross_topic_test'], [cross_topic_val, cross_topic_test],
                                         save_path, dataset_name)

    logging.info('summarizing cross data')
    # write some summaries for them too
    topic_stats_path = os.path.join(save_path, f'{dataset_name}_cross_topic_stats.json')
    summarize_dataset(original_data, train, cross_topic_val, cross_topic_test, cross_topic_av[0], cross_topic_av[1],
                      topic_stats_path)

    if cross_genre_test is not None:
        cross_genre_av = save_aa_and_av_dset(['cross_genre_val', 'cross_genre_test'],
                                             [cross_genre_val, cross_genre_test],
                                             save_path, dataset_name)
        genre_stats_path = os.path.join(save_path, f'{dataset_name}_cross_genre_stats.json')
        summarize_dataset(original_data, train, cross_genre_val, cross_genre_test, cross_genre_av[0], cross_genre_av[1],
                          genre_stats_path)
    logging.info('done')


def finalize_AV_dataset(original_data: Dict, train: Dict, val: Dict, test: Dict, av_val: Dict, av_test: Dict,
                        dataset_name, save_path):

    finalize_dataset(original_data, train, val, test, dataset_name, save_path)

    logging.info(f'writing the unique val and test sets.')

    for i, (name, dset) in enumerate(zip(['unique_val', 'unique_test'], [av_val, av_test])):
        if dset:
            av_fp = os.path.join(save_path, f'{dataset_name}_AV_{name}.csv')
            logging.info(f'sampling the non-author-overlapping AV {name} set')
            av = sample_av_dataset(dset)
            logging.info(f'writing the AV {name} set to {av_fp}')
            write_av_dataset(av, av_fp)

    logging.info('complete')

if __name__ == '__main__':
    import argparse
    from valla.dsets.loaders import get_av_dataset, get_aa_dataset
    # subsample a dataset iid w.r.t. data points
    # get command line args
    parser = argparse.ArgumentParser(description='Run a BertAA model from the command line')

    parser.add_argument('--file_path', type=str)
    parser.add_argument('--av', action='store_true')
    parser.add_argument('--n', type=int, default=1000)
    parser.set_defaults(av=False)

    args = parser.parse_args()

    # get the data
    logging.info(f'getting the original {"av" if args.av else "aa"} dataset from {args.file_path}')
    if args.av:
        data = get_av_dataset(args.file_path)
    else:
        data = get_aa_dataset(args.file_path)

    logging.info(f'there are {len(data)} samples in the original dataset')

    # now subsample to n samples
    logging.info(f'subsampling the dataset to {args.n} samples')
    smaller_data = random.sample(data, args.n)

    # now write the subsampled dataset
    logging.info(f'writing the new {"av" if args.av else "aa"} dataset')
    if args.av:
        write_av_dataset(smaller_data, args.file_path+'.sub')
    else:
        write_aa_dataset(list_dset_to_dict(smaller_data), args.file_path+'.sub')
