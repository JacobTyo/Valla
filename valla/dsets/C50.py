"""
modified from the script found here:
    https://github.com/bmurauer/authbench/blob/main/scripts/unify_c50.py

dataset access:
    https://archive.ics.uci.edu/ml/datasets/Reuter_50_50
"""
from sklearn.model_selection import train_test_split
from valla.utils.dataset_utils import finalize_dataset
import logging
from tqdm import tqdm
import os
import argparse
import random
import numpy as np
from glob import glob
from typing import List, Dict


def process_c50(pth, seed=0):
    processed_dir = os.path.join(pth, "processed")
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)
    raw_dir = os.path.join(pth, "raw")

    train = os.path.join(raw_dir, "C50train")
    test = os.path.join(raw_dir, "C50test")

    def read(subdir: str, author_ids: dict) -> List[Dict]:
        posts = {}
        authors = os.listdir(subdir)
        for j, author in enumerate(authors):
            author_dir = os.path.join(subdir, author)
            files = glob(author_dir + "/*.txt")
            for f in files:
                with open(f) as i_f:
                    text = i_f.read()
                    posts.setdefault(author_ids.setdefault(author, j), []).append(text)
        return posts, author_ids

    logging.info('getting train and test sets')
    auth_to_id = {}  # make sure author id's are consistent across train and test set
    train_and_eval_dict, auth_to_id = read(train, auth_to_id)
    test_dict, auth_to_id = read(test, auth_to_id)

    # make a dict of all data for stat tracking
    all_data = {}
    for data in [train_and_eval_dict, test_dict]:
        for k, v in data.items():
            for t in v:
                all_data.setdefault(k, []).append(t)

    # we need to split the train into a training and evaluation set
    train_and_eval_data = []
    for auth in train_and_eval_dict.keys():
        for text in train_and_eval_dict[auth]:
            train_and_eval_data.append([auth, text])

    logging.info(f'splitting the training data into train/eval sets')
    # now split into stratified train(60%)/val(20%)/test(20%) splits
    train_set, eval_set = train_test_split(train_and_eval_data, test_size=0.2, shuffle=True, random_state=seed,
                                                    stratify=[lbl for lbl, _ in train_and_eval_data])

    # now transform back to dicts
    train_dict = {}
    for auth, text in train_set:
        train_dict.setdefault(auth, []).append(text)

    val_dict = {}
    for auth, text in eval_set:
        val_dict.setdefault(auth, []).append(text)

    logging.info('finalizing')
    finalize_dataset(all_data, train_dict, val_dict, test_dict,
                     dataset_name='CCAT50', save_path=processed_dir)
    logging.info('done')


if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of the CCAT50 dataset')

    parser.add_argument('--dataset_path', type=str, default='/home/jtyo/data/Projects/'
                                                            'On_the_SOTA_of_Authorship_Verification/datasets/CCAT50')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    process_c50(args.dataset_path, args.seed)
