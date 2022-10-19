# dataset citation: "Authorship Attribution of Micro-Messages" (contact authors for dataset access)
from valla.utils.dataset_utils import finalize_dataset, dict_dset_to_list, list_dset_to_dict
from sklearn.model_selection import train_test_split
import argparse
import os
import logging
import random
import numpy as np

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    # get command line args

    parser = argparse.ArgumentParser(description='Get args for building train/test splits of the twitter dataset')

    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--dataset_save_path', type=str)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # dir to the dataset
    dataset_path = args.dataset_path
    dataset_save_path = args.dataset_save_path

    # now get all users and tweets
    logging.info(f'walking through {dataset_path}')
    data = {}
    user_id_to_label = {}
    label_counter = 0
    for root, dirs, files in os.walk(dataset_path):
        # print(f'{root}, {dirs}, {files}')
        # every dir represents a user, every file represents a list of tweets (one per line)
        for f in files:
            user_id = int(root.split('/')[-1].strip())
            logging.debug(f'reading {f} from {root}, with uid {user_id}')
            if user_id not in user_id_to_label:
                user_id_to_label[user_id] = label_counter
                label_counter += 1
            with open(os.path.join(root, f), 'r') as f_data:
                tweets = f_data.readlines()
                data.setdefault(user_id_to_label[user_id], []).extend(tweets)

    num_authors = len(list(data.keys()))
    num_texts = sum([len(v) for v in data.values()])
    logging.info(f'{num_authors} authors and {num_texts} tweets')

    # splitting all_data 80-10-10 iid
    logging.info('spliting data into train/val/test')
    list_data = dict_dset_to_list(data)
    train_set, eval_and_test_set = train_test_split(list_data, test_size=0.2, shuffle=True, random_state=args.seed,
                                                    stratify=[lbl for lbl, _ in list_data])
    eval_set, test_set = train_test_split(eval_and_test_set, test_size=0.5, shuffle=True, random_state=args.seed,
                                          stratify=[lbl for lbl, _ in eval_and_test_set])

    # finalize
    logging.info('finalizing')
    finalize_dataset(list_dset_to_dict(list_data),
                     list_dset_to_dict(train_set),
                     list_dset_to_dict(eval_set),
                     list_dset_to_dict(test_set),
                     'Twitter_micro',
                     args.dataset_save_path)
    logging.info('Finished!')
