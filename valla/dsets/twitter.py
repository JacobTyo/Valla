# dataset citation: https://archive.org/details/twitter_cikm_2010
# information about raw dataset format: https://ia600501.us.archive.org/32/items/twitter_cikm_2010/twitter_cikm_2010.pdf
from typing import Tuple
from valla.utils.dataset_utils import finalize_dataset, dict_dset_to_list, list_dset_to_dict
from sklearn.model_selection import train_test_split
import argparse
import os
import csv
import logging
import random
import numpy as np

logging.basicConfig(level=logging.INFO)


def open_data(tweets_path: str, label_map: dict = None) -> Tuple[dict, dict]:
    data = {}
    # do this to avoid mutable default arg
    build_new_label_map = True if label_map is None else False
    label_map = label_map if label_map is not None else {}
    user_counter = 0
    errored_line_count = 0
    missing_from_test = 0
    unique_tweets = set()
    duplicated_tweets = 0
    with open(tweets_path, 'r') as f:
        # format: UserID \t TweetID \t tweet \t create_datetime
        tweet_reader = csv.reader((line.replace('\0', '') for line in f), delimiter='\t')

        total_current_line = []

        for tweet_info in tweet_reader:

            if len(tweet_info) != 4:
                # have a better feel for errors, now just track total number
                # logging.warning(f'got line of wrong len: {len(tweet_info)}\n\t{tweet_info}')
                errored_line_count += 1

                for thing in tweet_info:
                    thing = thing.strip()
                    if len(thing) > 1:
                        total_current_line.append(thing)

                if len(total_current_line) > 4:
                    # trash and start over. . .
                    total_current_line = []
                    continue
                elif len(total_current_line) == 4:
                    tweet_info = total_current_line
                else:
                    continue

            try:
                user_id = int(tweet_info[0])
            except:
                # just drop it too. . .
                continue
            tweet = tweet_info[2]

            if tweet not in unique_tweets:
                unique_tweets.add(tweet)
            else:
                duplicated_tweets += 1
                continue

            if user_id not in label_map:
                if not build_new_label_map:
                    # logging.warning('a test label was not found in the label map')
                    missing_from_test += 1
                    continue
                label_map[user_id] = user_counter
                user_counter += 1

            data.setdefault(label_map[user_id], []).append(tweet)

    # ensure no duplicated tweets for each author (does not check if two authors have the same tweet)
    for k in list(data.keys()):
        data[k] = list(set(data[k]))

    if errored_line_count > 0:
        logging.warning(f'{errored_line_count} lines encountered an error')
    if missing_from_test > 0:
        logging.warning(f'{missing_from_test} examples from users who we deleted their data from in the '
                        f'training set (so no training data, they were dropped)')
    if duplicated_tweets > 0:
        logging.warning(f'{duplicated_tweets} examples were exact duplicates, removed.')

    return data, label_map


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
    
    train_tweets_path = os.path.join(dataset_path, 'training_set_tweets.txt')
    # we don't care about location data so ignoring this
    # train_users = os.path.join(dataset_path, 'training_set_authors.txt')
    test_tweets_path = os.path.join(dataset_path, 'test_set_tweets.txt')
    # we don't care about location data so ignoring this
    # test_users = os.path.join(dataset_path, 'test_set_authors.txt')

    logging.info('reading training data')
    # get the training data
    train_tweets, userid_to_label_map = open_data(train_tweets_path)
    # make sure we read the data right
    num_train_auths = len(list(train_tweets.keys()))
    if num_train_auths != 115886:
        logging.warning(f'The number of train authors ({num_train_auths}) does not match that published (115886)')

    # drop authors with less than 3 texts. . .
    dropped_auths = 0
    for auth in list(train_tweets.keys()):
        if len(train_tweets[auth]) < 3:
            map_key = [k for k, v in userid_to_label_map.items() if v == auth][0]
            del train_tweets[auth]
            del userid_to_label_map[map_key]
            dropped_auths += 1
    logging.warning(f'dropped {dropped_auths} auths due to too few tweets.')

    logging.info('reading testing data')
    # read test data
    # ensure that we use the same label mapping as the training set here
    test_tweets, _ = open_data(test_tweets_path, userid_to_label_map)
    num_test_auths = len(list(test_tweets.keys()))
    if num_test_auths != 5136:
        logging.warning(f'The number of test authors ({num_test_auths}) does not match that published (5136)')

    logging.info('splitting training into training and validation set')
    train_tweet_list = dict_dset_to_list(train_tweets)
    train_set, val_set = train_test_split(train_tweet_list, test_size=0.1, shuffle=True, random_state=args.seed,
                                          stratify=[lbl for lbl, _ in train_tweet_list])

    logging.info('finalizing and writing')
    # finalize 
    finalize_dataset(train_tweets,
                     list_dset_to_dict(train_set),
                     list_dset_to_dict(val_set),
                     test_tweets,
                     'Twitter',
                     args.dataset_save_path)
    logging.info('finished')



