# Data access: https://github.com/noa/naacl2021
#  https://github.com/bmurauer/reddit_corpora
from valla.utils.dataset_utils import finalize_dataset, dict_dset_to_list, list_dset_to_dict
from sklearn.model_selection import train_test_split
import logging
import argparse
import json
import random
import numpy as np


if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of the reddit dataset')

    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--dataset_save_path', type=str)
    parser.add_argument('--num_authors', type=int, default=25000)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # the file is huge so we need to get line by line, and I think we are only going to get a subset, not sure what else to do really.
    auth_id_to_int, auth_counter = {}, 0
    data = {}
    unique_texts = set([])
    duplicated_text_counter = 0
    with open(args.dataset_path, 'r') as reddit_file:
        for line in reddit_file:
            line_data = json.loads(line)
            author_id = line_data['author_id']
            if author_id not in auth_id_to_int:
                auth_id_to_int[author_id] = auth_counter
                auth_counter += 1
            texts = line_data['syms']
            for text in texts:
                if text not in unique_texts:
                    unique_texts.add(text)
                    data.setdefault(auth_id_to_int[author_id], []).append(text)
                else:
                    duplicated_text_counter += 1
            if auth_counter >= args.num_authors:
                break

    logging.info(f'{duplicated_text_counter} duplicate texts found')
    logging.info(f'{len(list(data.keys()))} authors collected')
    logging.info(f'{len(unique_texts)} texts collected')

    # finalize the data, maybe play with it a little to get bigger, but don't want too big, maybe 5gb?
    #    idk though, 5gb would take forever to train, perhhaps 1gb would be more like it. It'll be slow regardless. idk
    #    the dataset is already filtered to have only users with between 100 and 1000 posts, so for now I'm just going
    #    to take the first k author (I'm thinking k=25000)
    all_data = dict_dset_to_list(data)

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
                     'Reddit',
                     args.dataset_save_path)
