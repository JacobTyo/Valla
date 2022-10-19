"""
This script was adapted from:
    https://github.com/bmurauer/authbench/blob/main/scripts/unify_cmcc.py

Dataset Access
    Email authoros of: Creating and Using a Correlated Corpora to Glean Communicative Commonalities
    http://www.lrec-conf.org/proceedings/lrec2008/pdf/771_paper.pdf
"""
import os
import re
from glob import glob
from typing import Dict, Any
from valla.utils.dataset_utils import finalize_cross_dataset, list_dset_to_dict, auth_text_make_unique
from sklearn.model_selection import train_test_split

import pandas as pd
import argparse
import random
import numpy as np


def check_or_fix_dataset_typo(directory: str) -> None:
    """
    There is one typo in the dataset which might have not been corrected yet:
    there is one file 'Discussions/Correlated/S1D113.txt'
    Which is the only file in the corpus that does not comply to the
    naming convention explained in FileCodingSchemes3.doc.
    It should be called S1D1I3.txt with an upper case i instead of a digit one.
    This code was tested on CMCCData.zip with a md5 checksum of:
        157586057cf4ad3dc1876890e94373a5
    """
    wrong = os.path.join(directory, "Discussion", "Correlated", "S1D113.txt")
    right = os.path.join(directory, "Discussion", "Correlated", "S1D1I3.txt")

    if os.path.isfile(wrong):
        print("renaming " + wrong + " to " + right)
        os.rename(wrong, right)


def process_cmcc(pth: str) -> Dict[str, Dict[Any, Any]]:
    processed_dir = os.path.join(pth, "processed")
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)
    directory = os.path.join(pth, "raw")

    check_or_fix_dataset_typo(directory)

    train_posts, cross_topic_posts, cross_genre_posts = {}, {}, {}
    auth_to_id = {}
    auth_counter = 0
    categories = ["Blogs", "Chat", "Discussion", "Emails", "Essays", "Interviews"]
    train_categories, train_texts = ["Blogs", "Chat", "Emails"], 0
    cross_topic_categories, cross_topic_texts = ["Essays"], 0
    cross_genre_categories, cross_genre_texts = ["Discussion", "Interviews"], 0

    for category in categories:
        correlated_dir = os.path.join(directory, category, "Correlated")
        files = glob(correlated_dir + "/*.txt")
        pattern = re.compile(
            r"(?P<author>[A-Z]\d+)(?P<genre>[A-Z])\d+(?P<topic>[A-Z])\d+.txt"
        )

        for f in files:
            # the files are windows-1252-encoded.
            with open(f, "rb") as i_f:
                try:
                    text_raw = i_f.read().decode("cp1252")
                except Exception as e:
                    print(f)
                    raise e

            name = os.path.basename(f)
            match = pattern.match(name)
            if not match:
                raise ValueError("no match found for file: " + f)

            # we only need text_raw and match.groupdict()['author']
            if match.groupdict()['author'] not in auth_to_id:
                auth_to_id[match.groupdict()['author']] = auth_counter
                auth_counter += 1

            a = auth_to_id[match.groupdict()['author']]

            if category in train_categories:
                train_posts.setdefault(a, []).append(text_raw)
                train_texts += 1

            if category in cross_topic_categories:
                cross_topic_posts.setdefault(a, []).append(text_raw)
                cross_topic_texts += 1

            if category in cross_genre_categories:
                cross_genre_posts.setdefault(a, []). append(text_raw)
                cross_genre_texts += 1

    print(f'there are {train_texts} iid texts')
    print(f'there are {cross_topic_texts} cross topic texts')
    print(f'there are {cross_genre_texts} cross genre texts')

    return {
        'train': train_posts,
        'cross_topic': cross_topic_posts,
        'cross_genre': cross_genre_posts,
    }


if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of the cmcc dataset')

    parser.add_argument('--dataset_path', type=str, default='/home/jtyo/data/Projects/'
                                                            'On_the_SOTA_of_Authorship_Verification/datasets/CMCC')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    all_data = process_cmcc(args.dataset_path)

    # this gives us the training set, a cross topic set, and a cross genre set. This is realy three problems, so ideally
    # we will have: train, val, test, cross_topic_val, cross_topic_test, cross_genre_val, cross_genre_test
    # so make lists of the sets and split
    iid_data = []
    for auth, texts in all_data['train'].items():
        for text in texts:
            iid_data.append([auth, text])
    train_set, eval_and_test_set = train_test_split(iid_data, test_size=0.4, shuffle=True, random_state=args.seed,
                                                    stratify=[lbl for lbl, _ in iid_data])
    eval_set, test_set = train_test_split(eval_and_test_set, test_size=0.5, shuffle=True, random_state=args.seed,
                                                    stratify=[lbl for lbl, _ in eval_and_test_set])

    all_cross_topic = []
    for auth, texts in all_data['cross_topic'].items():
        for text in texts:
            all_cross_topic.append([auth, text])
    cross_topic_eval, cross_topic_test = train_test_split(all_cross_topic, test_size=0.5, shuffle=True,
                                                          random_state=args.seed,
                                                          stratify=[lbl for lbl, _ in all_cross_topic])

    all_cross_genre = []
    for auth, texts in all_data['cross_genre'].items():
        for text in texts:
            all_cross_genre.append([auth, text])
    cross_genre_eval, cross_genre_test = train_test_split(all_cross_genre, test_size=0.5, shuffle=True,
                                                          random_state=args.seed,
                                                          stratify=[lbl for lbl, _ in all_cross_genre])

    # now finalize the dataset.
    original_data = []
    for dset_name, dset in all_data.items():
        for auth, texts in dset.items():
            for text in texts:
                original_data.append([auth, text])

    save_path = os.path.join(args.dataset_path, 'processed')
    finalize_cross_dataset(original_data=list_dset_to_dict(original_data),
                           train=list_dset_to_dict(train_set),
                           val=list_dset_to_dict(eval_set),
                           test=list_dset_to_dict(test_set),
                           cross_topic_val=list_dset_to_dict(cross_topic_eval),
                           cross_topic_test=list_dset_to_dict(cross_topic_test),
                           cross_genre_val=list_dset_to_dict(cross_genre_eval),
                           cross_genre_test=list_dset_to_dict(cross_genre_test),
                           dataset_name='CMCC',
                           save_path=save_path)
