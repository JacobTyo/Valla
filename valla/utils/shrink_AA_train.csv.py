import os
import argparse
import logging
from tqdm import tqdm
from valla.dsets.loaders import get_aa_dataset
from valla.utils.dataset_utils import write_aa_dataset, list_dset_to_dict

if __name__ == '__main__':
    # get each text in the dataset and do one same author and one different author. . . at least??
    parser = argparse.ArgumentParser(description='Build a sampled av dataset from an aa dataset')

    parser.add_argument('--file_path', type=str)
    parser.add_argument('--txt_size_lim', type=int, default=None)
    parser.add_argument('--max_txts_per_auth', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    txt_size_lim = args.txt_size_lim
    max_txts_per_auth = args.max_txts_per_auth
    file_path = args.file_path

    logging.info(f'getting dataset from {file_path}')
    dset = list_dset_to_dict(get_aa_dataset(file_path))

    if max_txts_per_auth is not None:
        logging.info(f'keeping only {max_txts_per_auth} texts per author')
        for key in list(dset.keys()):
            if len(dset[key]) > max_txts_per_auth:
                dset[key] = dset[key][:max_txts_per_auth]

    if txt_size_lim is not None:
        logging.info(f'shrinking all txt samples to a maximum of {txt_size_lim} characters')
        for key in list(dset.keys()):
            for val_idx in range(len(dset[key])):
                dset[key][val_idx] = dset[key][val_idx][:txt_size_lim]

    save_path = file_path + '.sub'
    logging.info(f'saving to {save_path}')
    write_aa_dataset(dset, save_path)
