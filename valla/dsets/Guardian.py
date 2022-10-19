"""
Based on the script found here:
    https://github.com/bmurauer/authbench/blob/main/scripts/unify_guardian.py

Dataset access:
    https://www.kaggle.com/datasets/adityakharosekar2/guardian-news-articles
"""
import os
from glob import glob
import argparse
import random
import numpy as np
import logging
from typing import List, Dict
from sklearn.model_selection import train_test_split
from valla.utils.dataset_utils import finalize_cross_dataset, list_dset_to_dict, auth_text_make_unique, \
    dict_dset_to_list

logging.basicConfig(level=logging.INFO)


def process_guardian(directory: str) -> List[Dict[str, str]]:
    processed_dir = os.path.join(directory, "processed")
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)
    directory = os.path.join(directory, "raw")

    posts = []
    categories = os.listdir(directory)
    for category in categories:
        c_dir = os.path.join(directory, category)
        if not os.path.isdir(c_dir):
            continue
        authors = os.listdir(c_dir)
        for author in authors:
            author_dir = os.path.join(c_dir, author)
            files = glob(author_dir + "/*.txt")
            for f in files:
                # files are windows-encoded
                with open(f, "rb") as i_f:
                    try:
                        posts.append(
                            dict(
                                category=category,
                                author=author,
                                text_raw=i_f.read().decode("cp1252"),
                            )
                        )
                    except Exception as e:
                        print(f)
                        raise e
    return posts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of the Guardian dataset')

    parser.add_argument('--dataset_path', type=str, default='/home/jtyo/data/Projects/'
                                                            'On_the_SOTA_of_Authorship_Verification/datasets/CCAT50')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    logging.info(f'getting the guardian data')

    all_data = process_guardian(args.dataset_path)

    # we want a single training set, which will be Politics, World, and Society
    # Build an IID eval split - 20% of the training set

    # Build a cross topic split
    #   just pick society as test set

    # build a cross genra split
    #   make training everything except books, then test on books - what about for an evaluation set?!?!
    logging.info(f'building the different splits')
    original_data, all_iid = [], []
    all_cross_topic, all_cross_genre = [], []

    auth_id_normalize, auth_id_counter = {}, 0
    for post in all_data:
        if post['author'] not in auth_id_normalize:
            auth_id_normalize[post['author']] = auth_id_counter
            auth_id_counter += 1

        original_data.append([auth_id_normalize[post['author']], post['text_raw']])

        if post['category'] in ['Politics', 'Society', 'World']:
            all_iid.append([auth_id_normalize[post['author']], post['text_raw']])

        elif post['category'] in ['UK']:
            all_cross_topic.append([auth_id_normalize[post['author']], post['text_raw']])

        elif post['category'] in ['Books']:
            all_cross_genre.append([auth_id_normalize[post['author']], post['text_raw']])

        else:
            raise ValueError(f"the category of the post is unrecognized: {post['category']}")

    # get iid train/val/test sets
    # duplicate texts found, remove from all_iid
    all_iid = dict_dset_to_list(auth_text_make_unique(list_dset_to_dict(all_iid)))

    train_set, val_and_test_set = train_test_split(all_iid, test_size=0.4, shuffle=True, random_state=args.seed,
                                                       stratify=[lbl for lbl, _ in all_iid])
    val_set, test_set = train_test_split(val_and_test_set, test_size=0.5, shuffle=True, random_state=args.seed,
                                             stratify=[lbl for lbl, _ in val_and_test_set])

    # get cross-topic val/test sets
    cross_topic_val, cross_topic_test = train_test_split(all_cross_topic, test_size=0.5, shuffle=True,
                                                         random_state=args.seed,
                                                         stratify=[lbl for lbl, _ in all_cross_topic])

    # get cross-genre val/test sets
    cross_genre_val, cross_genre_test = train_test_split(all_cross_genre, test_size=0.5, shuffle=True,
                                                         random_state=args.seed,
                                                         stratify=[lbl for lbl, _ in all_cross_genre])


    logging.info(f'finalizing and saving')
    save_path = os.path.join(args.dataset_path, 'processed')
    finalize_cross_dataset(original_data=list_dset_to_dict(original_data),
                           train=list_dset_to_dict(train_set),
                           val=list_dset_to_dict(val_set),
                           test=list_dset_to_dict(test_set),
                           cross_topic_val=list_dset_to_dict(cross_topic_val),
                           cross_topic_test=list_dset_to_dict(cross_topic_test),
                           cross_genre_val=list_dset_to_dict(cross_genre_val),
                           cross_genre_test=list_dset_to_dict(cross_genre_test),
                           dataset_name='guardian',
                           save_path=save_path)
