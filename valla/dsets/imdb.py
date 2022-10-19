# dataset access: https://umlt.infotech.monash.edu/?page_id=266
import os
import random
import logging
import argparse
from valla.utils import dataset_utils
import numpy as np


def get_imdb_as_dict(dataset_path: str) -> dict:
    """

    Parameters
    ----------
    dataset_path: The path to the IMDb62.txt file.

    Returns
    -------
    A dictionary representation of the IMDb62 dataset with the key being the author id and the value being
    a list of texts.

    """

    # get the dataset as a dict, with key being author id and value being a list of contents
    raw_data = {}

    # transform user_id's as well
    label_transformer = {}
    label_count = 0

    with open(dataset_path, 'r') as f:

        lines = f.readlines()

        for line in lines:

            line = line.split('\t')

            # we want to make sure that splitting on tab is the right thing, so check len of resultant object
            assert len(line) == 6, 'The split line, from the imdb62 dataset, has not given in the right num of objects'

            # get the author
            user_id = line[1]

            # get the title and content
            text = line[-2] + ' ' + line[-1]

            if user_id not in label_transformer.keys():
                label_transformer[user_id] = label_count
                label_count += 1

            # change user_id to incrementing int
            user_id = str(label_transformer[user_id])

            if user_id not in raw_data.keys():
                raw_data[user_id] = [text]
            else:
                raw_data[user_id].append(text)

        # we are seeing issues with duplicates, remove
        for auth, texts in raw_data.items():
            raw_data[auth] = list(set(texts))

    return raw_data


if __name__ == "__main__":
    # This file splits the IMDb62 dataset into train/validation/test splits, and stores them in the common data format.
    # This file is deterministic and should be used on the dataset downloaded here:
    #   https://www.dropbox.com/s/np1u1hl343gd73m/imdb62.zip

    # set logging level
    logging.basicConfig(level=logging.INFO)

    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of IMDB dataset')

    parser.add_argument('--dataset_path', type=str, default='datasets/imdb/raw/imdb62/imdb62.txt',
                        help='The path to the unzipped IMDb62 dataset.')
    parser.add_argument('--output_path', type=str, default='datasets/imdb/processed/imdb62',
                        help='Where to save the normalized version of the dataset.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # check input file and output path
    assert os.path.isfile(args.dataset_path), '--dataset_path must point to the imdb62.txt file.'
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    assert os.path.isdir(args.output_path), '--output_path must be a directory'

    # read the csv file. The dataset isn't very big so keep everything in memory
    logging.info(f'Getting original dataset at {args.dataset_path}')
    data = get_imdb_as_dict(args.dataset_path)

    # now split this into a train, validation, and test dict (taking first 700 training,
    #   then next 150 for validation, and last 150 for test is fine)
    logging.info(f'Splitting dataset into training, validation, and test')
    train_data, val_data, test_data = {}, {}, {}
    for k, v in data.items():
        train_data[k] = v[:600]
        val_data[k] = v[600:800]
        test_data[k] = v[800:]

    # write this normalized version of the dataset.
    train_fp = os.path.join(args.output_path, 'imdb62_train.csv')
    aa_val_fp = os.path.join(args.output_path, 'imdb62_AA_val.csv')
    aa_test_fp = os.path.join(args.output_path, 'imdb62_AA_test.csv')
    logging.info(f'writing the training, AA validation, and AA testing splits to {train_fp}, {aa_val_fp}, '
                 f'and {aa_test_fp} respectively')

    for aa_dset, save_fp in [(train_data, train_fp), (val_data, aa_val_fp), (test_data, aa_test_fp)]:
        dataset_utils.write_aa_dataset(aa_dset, save_fp)

    # now build the AV validation and test sets.
    # the methodology is for every text in the val and test sets, build one same and one different author sample.
    av_val = dataset_utils.sample_av_dataset(val_data)
    av_test = dataset_utils.sample_av_dataset(test_data)

    # write the av val and test set
    av_val_fp = os.path.join(args.output_path, 'imdb62_AV_val.csv')
    av_test_fp = os.path.join(args.output_path, 'imdb62_AV_test.csv')
    logging.info(f'writing the AV validation and AV testing splits to {av_val_fp} '
                 f'and {av_test_fp} respectively')

    for av_dset, av_fp in [(av_val, av_val_fp), (av_test, av_test_fp)]:
        dataset_utils.write_av_dataset(av_dset, av_fp)

    # also take advantage of this opportunity to do some dataset analysis and write to file.
    save_path = os.path.join(args.output_path, 'imdb62_stats.json')
    logging.info(f'calculating data statistics and writing to {save_path}')
    dataset_utils.summarize_dataset(data, train_data, val_data, test_data, av_val, av_test, save_path)

    logging.info('complete')
