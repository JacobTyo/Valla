import csv
import random
import numpy as np
import argparse
import logging
from valla.dsets.loaders import get_aa_dataset, aa_as_pandas, get_av_dataset
from valla.utils.dataset_utils import write_aa_dataset, write_av_dataset

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get args for making a small version of a data file (for testing)')
    parser.add_argument('--input_data', type=str, default='/home/jtyo/data/CCAT50/processed/CCAT50_AA_val.csv')
    parser.add_argument('--output_path', type=str, default='/home/jtyo/testing.csv')
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--av', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.av:
        logging.info(f'getting AV dset from {args.input_data} and shrinking to {args.size} samples')
        data = get_av_dataset(args.input_data)
        logging.info(f'sampling the dataset to {args.size} samples')
        data = random.sample(data, args.size)
        logging.info(f'writing av dataset of size {args.size} to {args.output_path}')
        write_av_dataset(data, args.output_path)
        logging.info('done')

    else:
        logging.info(f'getting AA dset from {args.input_data} and shrinking to {args.size} samples')
        data = aa_as_pandas(get_aa_dataset(args.input_data))
        data = data.sample(n=args.size)

        dset = {}
        id_normalize, id_counter = {}, 0
        for auth, txt in zip(data['labels'], data['text']):
            if auth not in id_normalize:
                id_normalize[auth] = id_counter
                id_counter += 1
            dset.setdefault(id_normalize[auth], []).append(txt)

        logging.info(f'writing debugging dataset to {args.output_path}')
        write_aa_dataset(dset, args.output_path)
        logging.info('done')
