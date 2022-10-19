import os
import argparse
import logging
from tqdm import tqdm
from valla.dsets.loaders import get_aa_dataset
from valla.utils.dataset_utils import write_av_dataset, sample_av_dataset, list_dset_to_dict

if __name__ == '__main__':
    # get each text in the dataset and do one same author and one different author. . . at least??
    parser = argparse.ArgumentParser(description='Build a sampled av dataset from an aa dataset')

    parser.add_argument('--file_path', type=str)
    parser.add_argument('--txt_size_lim', type=int, default=None)
    parser.add_argument('--dont_check_duplicates', action='store_true')
    parser.add_argument('--sample_once', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    logging.info(f'getting dataset from {args.file_path}')
    dset = list_dset_to_dict(get_aa_dataset(args.file_path))
    # some null byte issue
    logging.info('getting rid of null bytes and short texts')
    for k, v in dset.items():
        new_v = []
        for t in v:
            if len(t) > 50:
                t = t.replace('\0', '')
                new_v.append(t)
            else:
                logging.debug(f'removing a text of len {len(t)}')
        dset[k] = new_v

    logging.info(f'sampling an av version')
    av_dset = sample_av_dataset(dset, txt_size_lim=args.txt_size_lim)
    first_sample_len = len(av_dset)
    logging.info(f'sampled {first_sample_len} points')

    if not args.sample_once:
        logging.info(f'sampling again')
        # the seed isn't reset so sampling again, then make a set

        av_dset2 = sample_av_dataset(dset, txt_size_lim=args.txt_size_lim)
        second_sample_len = len(av_dset2)
        logging.info(f'sampled {second_sample_len} points')
        if args.dont_check_duplicates:
            logging.info('skipping duplicate checking')
            av_dset.extend(av_dset2)
        else:
            logging.info('making data unique')
            for sample in tqdm(av_dset2, desc=f'uniquify'):
                if sample not in av_dset:
                    av_dset.append(sample)
        final_len = len(av_dset)
        logging.info(f'final dset_len: {final_len}')
        logging.info(f'dropped {first_sample_len+second_sample_len-final_len} samples')
    save_path = args.file_path.split('train.csv')[0] + 'AV_train.csv'
    logging.info(f'saving to {save_path}')
    write_av_dataset(av_dset, save_path)
