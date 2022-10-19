# this is the dataset used in "Justifying Recommendations using Distantly-Labeled Reviews and
#  Fine-Grained Aspects" (https://aclanthology.org/D19-1018.pdf)
from valla.utils.dataset_utils import finalize_cross_dataset, dict_dset_to_list, list_dset_to_dict, \
    write_aa_dataset, write_av_dataset
from valla.dsets.loaders import get_aa_dataset, get_av_dataset
from sklearn.model_selection import train_test_split
import logging
import argparse
import os
from tqdm import tqdm
import json
import pickle
from sys import getsizeof
import numpy as np
import random


if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of the blogs dataset')

    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--dataset_save_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--min_texts', type=int, default=10)
    parser.add_argument('--max_texts', type=int, default=10)
    parser.add_argument('--check', action='store_true')

    args = parser.parse_args()

    logging.warning('make sure to run this script with --check after processing, '
                    'as problems with empty texts were encountered.')

    if args.check:

        # get the dataset
        if 'av' in args.dataset_path.split('/')[-1].lower():
            dset = get_av_dataset(args.dataset_path)
            # just check for weird stuff. . .
            dropable = 0
            empty = 0
            good_data = []
            for lbl, txt1, txt2 in dset:
                if len(txt1) < 50 or len(txt2) < 50:
                    dropable += 1
                    continue
                good_data.append([lbl, txt1, txt2])
            logging.info(f'{dropable} texts found less than 50 chars')
            logging.info(f'{empty} texts found empty')

            logging.info(f'rewriting dataset')
            write_av_dataset(good_data, args.dataset_path)

        else:
            dset = list_dset_to_dict(get_aa_dataset(args.dataset_path))
            # just check for weird stuff. . .
            dropable = 0
            empty = 0
            good_data = {}
            dont_write = False
            for lbl, txts in dset.items():
                this_auth_drop = 0
                for txt in txts:
                    if len(txt) < 50:
                        dropable += 1
                        this_auth_drop += 1
                        continue
                    good_data.setdefault(lbl, []).append(txt)

                if this_auth_drop > 9:
                    logging.warning('youve got problems bud')
                    dont_write = True

            logging.info(f'{dropable} texts found less than 50 chars')
            logging.info(f'{empty} texts found empty')

            if dont_write:
                logging.error('whole authors found that need dropped. . . not writing')
            else:
                logging.info(f'rewriting dataset')
                write_aa_dataset(good_data, args.dataset_path)

        exit(0)

    dataset_path = args.dataset_path
    dataset_save_path = args.dataset_save_path
    seed = args.seed

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Example review:
    # {
    # "image": ["https://images-na.ssl-images-amazon.com/images/I/71eG75FTJJL._SY88.jpg"],
    # "overall": 5.0,
    # "vote": "2",
    # "verified": True,
    # "reviewTime": "01 1, 2018",
    # "reviewerID": "AUI6WTTT0QZYS",
    # "asin": "5120053084",
    # "style": {
    # 	"Size:": "Large",
    # 	"Color:": "Charcoal"
    # 	},
    # "reviewerName": "Abbey",
    # "reviewText": "I now have 4 of the 5 available colors of this shirt... ",
    # "summary": "Comfy, flattering, discreet--highly recommended!",
    # "unixReviewTime": 1514764800
    # }

    # Categories
    # All_Beauty_5.json              Books_5.json                        Gift_Cards_5.json                 Magazine_Subscriptions_5.json  Prime_Pantry_5.json
    # amazon_5_links.txt             CDs_and_Vinyl_5.json                Grocery_and_Gourmet_Food_5.json   Movies_and_TV_5.json           Software_5.json
    # AMAZON_FASHION_5.json          Cell_Phones_and_Accessories_5.json  Home_and_Kitchen_5.json           Musical_Instruments_5.json     Sports_and_Outdoors_5.json
    # Appliances_5.json              Clothing_Shoes_and_Jewelry_5.json   Industrial_and_Scientific_5.json  Office_Products_5.json         Tools_and_Home_Improvement_5.json
    # Arts_Crafts_and_Sewing_5.json  Digital_Music_5.json                Kindle_Store_5.json               Patio_Lawn_and_Garden_5.json   Toys_and_Games_5.json
    # Automotive_5.json              Electronics_5.json                  Luxury_Beauty_5.json              Pet_Supplies_5.json            Video_Games_5.json

    # make an iid split and one with held out topics
    if not os.path.isfile('train_dump.pkl') or not args.cache:
        logging.info('starting from the raw data.')
        test_topics = ['CDs_and_Vinyl', 'Appliances', 'All_Beauty']

        # just group by author
        train_data = {}
        cross_topic_data = {}
        author_id_map, auth_counter = {}, 0

        # a directory of jsonl file
        for directory, subdirectories, files in tqdm(os.walk(dataset_path)):
            for file in tqdm(files):
                if 'json' not in file:
                    continue
                logging.info(f'processing {file}')
                with open(os.path.join(directory, file), 'r') as f:
                    for line in f.readlines():
                        line = json.loads(line)
                        amazon_author_id = line['reviewerID']
                        if amazon_author_id not in author_id_map:
                            author_id_map[amazon_author_id] = auth_counter
                            auth_counter += 1
                        author_id = author_id_map[amazon_author_id]
                        review_summary = line['summary'] if 'summary' in line else ''
                        if 'reviewText' not in line:
                            continue
                        review_text = line['reviewText']
                        total_review = review_summary + ' ' + review_text

                        if len(total_review) < 50:
                            continue
                        total_review = total_review[:100000].replace('\0', '').encode('utf-8').decode('utf-8')

                        # assign to the proper group
                        if any(x in file for x in test_topics):
                            cross_topic_data.setdefault(author_id, set([])).add(total_review)
                        else:
                            train_data.setdefault(author_id, set([])).add(total_review)
        print('pickling the data for later')
        with open('train_dump.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        with open('cross_dump.pkl', 'wb') as f:
            pickle.dump(cross_topic_data, f)
    else:
        logging.info('starting from the pickled files.')
        with open('train_dump.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('cross_dump.pkl', 'rb') as f:
            cross_topic_data = pickle.load(f)

    logging.info(f'filtering for author with >= {args.min_texts} texts')
    # filter it to keep only authors that have args.min_texts or more texts
    for auth in tqdm(list(train_data.keys())):
        if len(train_data[auth]) < args.min_texts or len(train_data[auth]) > args.max_texts:
            del train_data[auth]
    for auth in tqdm(list(cross_topic_data.keys())):
        if len(cross_topic_data[auth]) < args.min_texts or len(cross_topic_data[auth]) > args.max_texts:
            del cross_topic_data[auth]

    print('the dataset has the following number of texts')
    print(sum([len(v) for v in train_data.values()]))
    print(sum([len(v) for v in cross_topic_data.values()]))
    logging.info(f'the size of the training and cross topic data is:')
    logging.info(f'\t{getsizeof(train_data)/1024/1024} MB')
    logging.info(f'\t{getsizeof(cross_topic_data)/1024/1024} MB')

    # now split the train into a train (90%), val (10%), and test (10%) set
    org_train_list = dict_dset_to_list(train_data)
    train_list, val_and_test_list = train_test_split(org_train_list, test_size=0.4, shuffle=True,
                                                     random_state=args.seed,
                                                     stratify=[lbl for lbl, _ in org_train_list])

    val_list, test_list = train_test_split(val_and_test_list, test_size=0.5, shuffle=True, random_state=args.seed,
                                           stratify=[lbl for lbl, _ in val_and_test_list])

    # now split the cross topic data into a val and test
    cross_topic_list = dict_dset_to_list(cross_topic_data)
    val_cross_topic, test_cross_topic = train_test_split(cross_topic_list, test_size=0.5, shuffle=True,
                                                         random_state=args.seed,
                                                         stratify=[lbl for lbl, _ in cross_topic_list])

    # now finalize the dataset and we good to go
    finalize_cross_dataset(original_data=list_dset_to_dict(org_train_list),
                           train=list_dset_to_dict(train_list),
                           val=list_dset_to_dict(val_list),
                           test=list_dset_to_dict(test_list),
                           cross_topic_val=list_dset_to_dict(val_cross_topic),
                           cross_topic_test=list_dset_to_dict(test_cross_topic),
                           dataset_name='amazon',
                           save_path=args.dataset_save_path)
