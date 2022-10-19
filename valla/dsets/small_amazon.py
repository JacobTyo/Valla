# used to create very small testing dataset
from valla.utils.dataset_utils import finalize_cross_dataset, dict_dset_to_list, list_dset_to_dict, \
    write_aa_dataset, write_av_dataset
from valla.dsets.loaders import get_aa_dataset, get_av_dataset
from sklearn.model_selection import train_test_split
import logging
import argparse
import os
from tqdm import tqdm
import pandas as pd
import json
import pickle
import random
from bs4 import BeautifulSoup
from sys import getsizeof
import numpy as np


def clean_doc(doc):
    return BeautifulSoup(doc, 'html.parser').get_text().encode('utf-8').decode('utf-8')


if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of the blogs dataset')

    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_path = args.dataset_path

    train_data = []
    test_data = []

    data_panda = pd.read_csv(dataset_path, sep='\t')
    for idx in tqdm(range(data_panda.review.shape[0])):
        temp = data_panda.review[idx].split('$$$')
        doc1 = clean_doc(temp[0])
        doc2 = clean_doc(temp[1])

        if random.random() < 0.2:
            # test point
            test_data.append([data_panda.sentiment[idx], doc1, doc2])
        else:
            # train point
            train_data.append([data_panda.sentiment[idx], doc1, doc2])

    write_av_dataset(train_data, os.path.join(os.path.dirname(dataset_path), 'ama_test_AV_train.csv'))
    write_av_dataset(test_data, os.path.join(os.path.dirname(dataset_path), 'ama_test_AV_test.csv'))
