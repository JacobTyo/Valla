# Email jacob.tyo@gmail.com for a download link to this dataset.
import random
import numpy as np
import os
import argparse
import logging
import json
from tqdm import tqdm
import textacy.preprocessing as tp
from typing import List
from valla.utils.dataset_utils import finalize_AV_dataset, list_dset_to_dict, dict_dset_to_list
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)


def normalize_text(doc: str) -> str:
    doc = tp.normalize.unicode(doc)
    doc = tp.normalize.whitespace(doc)
    doc = tp.normalize.quotation_marks(doc)
    return doc


def check_overlap(auths0: List, auths1: List, set_names: str = '') -> None:
    overlap = 0
    for a in auths0:
        if a in auths1:
            logging.warning(f'{a} found in both the {set_names} set')
            overlap += 1
    if overlap > 0:
        logging.warning(f'there were {overlap} authors in both the {set_names} set')


def check_auth_text_count(data_set, id_map=None):
    _num_texts = 0
    if id_map:
        logging.debug('check_auth_text_count: using passed in id map')
        _auth_to_id = id_map
    else:
        logging.debug('check_auth_text_count: creating new id map')
        _auth_to_id = {}

    id_counter = 0
    _data_set = {}
    for auth, texts in data_set.items():
        text_set = set(texts)
        if len(text_set) >= 2:
            if (auth not in _auth_to_id) and not id_map:
                _auth_to_id[auth] = id_counter
                id_counter += 1

            _data_set[_auth_to_id[auth]] = list(text_set)
            _num_texts += len(text_set)
    return _data_set, _num_texts, _auth_to_id


if __name__ == '__main__':

    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of IMDB dataset')

    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    # starting from our train/val/test splits - all single-author english texts who's authors have >= 2 texts,
    # split iid between train/val/test
    train_pth = os.path.join(args.dataset_path, 'raw', 'train')
    val_pth = os.path.join(args.dataset_path, 'raw', 'val')
    test_pth = os.path.join(args.dataset_path, 'raw', 'test')

    train_data, val_data, test_data = {}, {}, {}

    # auth_to_id, id_counter = {}, 0

    # the data structure is files in folders, folder name represents author, so for each of the pths,
    # get a dict key'd on authors with values being a list of texts
    for data_pth, data_dict in zip([train_pth, val_pth, test_pth], [train_data, val_data, test_data]):
        for root, dirs, files in tqdm(os.walk(data_pth), desc=data_pth.split('/')[-1]):
            auth_name = list(root.split('/'))[-1]
            if auth_name in ['train', 'val', 'test']:
                # skip the base dirs
                continue
            # doing this in next step to avoid later processing problems.
            # if auth_name not in auth_to_id:
            #     auth_to_id[auth_name] = id_counter
            #     id_counter += 1

            for file in files:

                with open(os.path.join(root, file), 'r') as f:
                    try:
                        text = ' '.join(f.readlines())
                    except BaseException:
                        logging.warning(f'dropping text: {os.path.join(auth_name, file)}')

                    if len(text.strip().split(' ')) < 101:
                        logging.warning(f'dropping short text: {os.path.join(auth_name, file)}')
                    else:
                        data_dict.setdefault(auth_name, []).append(text)

    # make sure each author has at least two texts
    train_data, num_train_texts, auth_to_id = check_auth_text_count(train_data)
    val_data, num_val_texts, val_auth_to_id = check_auth_text_count(val_data)
    test_data, num_test_texts, test_auth_to_id = check_auth_text_count(test_data)

    for dset, num_texts, name in [[train_data, num_train_texts, 'train'],
                                  [val_data, num_val_texts, 'val'],
                                  [test_data, num_test_texts, 'test']]:
        logging.info(f"{num_texts} texts and {len(list(dset.keys()))} authors in the {name} set ")


    # just make sure lol
    train_auths = list(train_data.keys())
    val_auths = list(val_data.keys())
    test_auths = list(test_data.keys())
    check_overlap(train_auths, val_auths, 'train and val')
    check_overlap(train_auths, test_auths, 'train and test')
    check_overlap(test_auths, val_auths, 'val and test')

    # so now we can save both the aa and av datasets. However, here we need to be a little careful.
    #  The AA dataset needs to be all known authors - so we need a held-out set from the training set for this
    #  The AV dataset needs to be unknown authors, this should be fine under our current setup though.
    train_samples = dict_dset_to_list(train_data)
    train_samples, aa_val_and_test_samples = train_test_split(train_samples, test_size=0.4, random_state=args.seed,
                                                              shuffle=True,
                                                              stratify=[auth for auth, _ in train_samples])
    val_samples, test_samples = train_test_split(aa_val_and_test_samples, test_size=0.5, random_state=args.seed,
                                                 shuffle=True)
    # This fails as the authors don't have enough texts, for now just don't stratify after getting the eval/test sets
    #, stratify=[auth for auth, _ in aa_val_and_test_samples])

    # now save the datasets
    save_path = os.path.join(args.dataset_path, 'processed')
    finalize_AV_dataset(original_data=train_data,
                        train=list_dset_to_dict(train_samples),
                        val=list_dset_to_dict(val_samples),
                        test=list_dset_to_dict(test_samples),
                        av_val=val_data,
                        av_test=test_data,
                        dataset_name='gutenberg',
                        save_path=save_path)

    # save the auth_id dict just in case
    for id_map, name in zip([auth_to_id, val_auth_to_id, test_auth_to_id], ['train', 'val', 'test']):
        id_to_auth = {}
        for k, v in id_map.items():
            id_to_auth[v] = k
        with open(os.path.join(save_path, f'{name}_id_to_auth.json'), 'w') as f:
            json.dump(id_to_auth, f, sort_keys=True, indent=4)
