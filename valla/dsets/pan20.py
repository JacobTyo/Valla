# Data access: https://pan.webis.de/data.html#pan20-authorship-verification-large
#  Specific files used (both train and test sets):
#    - PAN20-Authorship-Verification
#    - PAN20-Authorship-Verification (Large)
#    - PAN21-Authorship-Verification
from valla.utils.dataset_utils import finalize_dataset, list_dset_to_dict, dict_dset_to_list, \
    write_av_dataset
from sklearn.model_selection import train_test_split
import logging
import os
import argparse
import json
import random
import numpy as np

logging.basicConfig(level=logging.INFO)
auth_id_transformer = {}
id_counter = 0


def get_jsonl_file(file_path):
    """
    """
    id_to_auths = {}
    # need to get the author id's first from the -truth file, matched by id
    truth_file = file_path.split('.jsonl')[0] + '-truth.jsonl'
    with open(truth_file, 'r') as f:
        for line in f:
            line_data = json.loads(line)
            id_to_auths[line_data['id']] = line_data['authors']

    raw_data = []

    with open(file_path, 'r') as f:

        for line in f:
            data_point = json.loads(line)
            _id = data_point['id']
            text0, text1 = data_point['pair']
            auth0, auth1 = id_to_auths[_id]
            raw_data.append([auth0, auth1, text0, text1])

    return raw_data


def normalize_pairs(pairs):
    normalized = []
    for auth0, auth1, text0, text1 in pairs:
        normalized.append([1 if auth0 == auth1 else 0, text0, text1])
    return normalized


def pairs_to_dict(raw_data):
    global id_counter
    global auth_id_transformer
    data = {}
    for auth0, auth1, text0, text1 in raw_data:
        for a, t in zip([auth0, auth1], [text0, text1]):
            if a not in auth_id_transformer:
                auth_id_transformer[a] = id_counter
                id_counter += 1
            data.setdefault(auth_id_transformer[a], []).append(t)

    # make sure unique
    for author, ts in data.items():
        data[author] = list(set(ts))

    return data


def merge_pan_large_small(large_small_dir):
    logging.warning('!!!!!!!!This action is not advised. For better comparability, '
                    'stick with using either the lare or small training set.!!!!!!!!!!!!!')
    large = os.path.join(large_small_dir, 'pan20-authorship-verification-training-large.jsonl')
    large_truth = os.path.join(large_small_dir, 'pan20-authorship-verification-training-large-truth.jsonl')
    small = os.path.join(large_small_dir, 'pan20-authorship-verification-training-small.jsonl')
    small_truth = os.path.join(large_small_dir, 'pan20-authorship-verification-training-small-truth.jsonl')

    all_train_truth = {}
    duplicated_ids = []
    small_count = 0
    for tf, tf_name in [(large_truth, 'large'), (small_truth, 'small')]:
        with open(tf, 'r') as f:
            for line in f:
                if tf_name == 'small':
                    small_count += 1
                line_data = json.loads(line)
                if line_data['id'] in all_train_truth:
                    duplicated_ids.append(line_data['id'])
                else:
                    all_train_truth[line_data['id']] = line_data

    logging.info(f'there were {len(duplicated_ids)} of the same samples in the small and large training sets')
    logging.info(f'there were {small_count} samples in the small dataset')

    all_train = []
    for tf in [large, small]:
        with open(tf, 'r') as f:
            for line in f:
                line_data = json.loads(line)
                if line_data['id'] in all_train_truth:
                    all_train.append(line_data)

    # now write the files and be done
    _all = os.path.join(large_small_dir, 'pan20-authorship-verification-training-all.jsonl')
    all_truth = os.path.join(large_small_dir, 'pan20-authorship-verification-training-all-truth.jsonl')
    with open(all_truth, 'w') as f:
        for _id, data in all_train_truth.items():
            f.write(json.dumps(data) + '\n')

    with open(_all, 'w') as f:
        for d in all_train:
            f.write(json.dumps(d)+'\n')

    logging.info('done combining files')


if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of the blogs dataset')

    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--dset_size', type=str, default='small')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--get_orig_train', action='store_true')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    AUTH_ID_TRANSFORM = {}
    ID_COUNTER = 0

    # if args.combine:
    #     # just combine the texts then exit
    #     merge_pan_large_small(args.train_path)
    #     exit(0)

    # get the blogs dataset as a dictionary
    logging.info(f'getting the train dataset file from {args.train_path}')
    av_train_pairs = get_jsonl_file(args.train_path)

    if args.get_orig_train:
        # just write the normalized train pairs
        save_path = os.path.join(os.path.dirname(os.path.dirname(args.train_path)), 'processed', args.dset_size, 'pan20_AV_train.csv')
        logging.info(f'saving the av train set to {save_path}')
        write_av_dataset(normalize_pairs(av_train_pairs), save_path)
        logging.info('finished')
        exit(0)

    logging.info(f'getting the test dataset file form {args.test_path}')
    av_test_pairs = get_jsonl_file(args.test_path)

    av_test_set = normalize_pairs(av_test_pairs)

    # now we will keep these pairs as the AV sets, but we will create dicts for the AA sets
    aa_train_dict = pairs_to_dict(av_train_pairs)
    test_set = pairs_to_dict(av_test_pairs)

    # some authors only have one sample, so force them into the training set
    forced_train_authors, remaining_authors = {}, {}
    for auth, texts in aa_train_dict.items():
        if len(texts) < 2:
            forced_train_authors[auth] = texts
        else:
            remaining_authors[auth] = texts

    # now split test into test and eval
    aa_train = dict_dset_to_list(remaining_authors)

    train_set, val_set = train_test_split(aa_train, test_size=0.2, shuffle=True, random_state=args.seed,
                                          stratify=[lbl for lbl, _ in aa_train])

    # now add back in the authors and texts that we had to pull out for the training set
    for auth, texts in forced_train_authors.items():
        for text in texts:
            train_set.append([auth, text])

    save_path = os.path.join(os.path.dirname(args.train_path), 'processed', args.dset_size)
    finalize_dataset(list_dset_to_dict(train_set), list_dset_to_dict(train_set), list_dset_to_dict(val_set), test_set,
                     dataset_name='pan20', save_path=save_path)

    # manually overwrite the test av set because the pairs are provided.
    overwrite_path = os.path.join(save_path, 'pan20_AV_test.csv')
    logging.info(f'overwriting the test set with the predefined one at {overwrite_path}')
    write_av_dataset(av_test_set, overwrite_path)
