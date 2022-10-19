# Dataset access: https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus
from valla.dsets.blogs import process_blogs_xmls
from valla.utils.dataset_utils import finalize_dataset, dict_dset_to_list, list_dset_to_dict
from sklearn.model_selection import train_test_split
import logging
import argparse
import random
import numpy as np


if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of the blogs dataset')

    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--dataset_save_path', type=str)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # get the blogs dataset as a dictionary
    logging.info(f'getting the original dataset file from {args.dataset_path}')
    data, author_counts, _, _, _, _ = process_blogs_xmls(args.dataset_path)

    # now just finalize an  AV dataset with these given only the authors with more than 4 texts?
    selected_auths = []
    total_texts = 0
    for auth, count in author_counts:
        if count >= 5:
            selected_auths.append(auth)
            total_texts += count

    logging.info(f'selected {len(selected_auths)} authors and {total_texts} texts')

    selected_data = {}
    for sauth in selected_auths:
        selected_data[sauth] = data[sauth]

    logging.info('splitting data into sets ')

    all_data = dict_dset_to_list(selected_data)

    train_set, eval_and_test_set = train_test_split(all_data, test_size=0.4, shuffle=True, random_state=args.seed,
                                                    stratify=[lbl for lbl, _ in all_data])
    aa_eval, aa_test = train_test_split(eval_and_test_set, test_size=0.5, shuffle=True, random_state=args.seed,
                                        stratify=[lbl for lbl, _ in eval_and_test_set])
    logging.info('finalizing')
    # original_data: Dict, train: Dict, val: Dict, test: Dict, dataset_name, save_path
    finalize_dataset(list_dset_to_dict(all_data),
                     list_dset_to_dict(train_set),
                     list_dset_to_dict(aa_eval),
                     list_dset_to_dict(aa_test),
                     'BlogsAV',
                     args.dataset_save_path)
