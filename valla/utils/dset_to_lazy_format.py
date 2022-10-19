# need to tranform from csv of auth_id, text to tsv of text, auth_id
from valla.dsets.loaders import get_aa_dataset
import argparse
import csv
import logging

logging.basicConfig(level=logging.DEBUG)


def normalize_string(str):
    return str.replace("\t", " ").replace('\n', ' ')


if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--test_set', action='store_true')
    args = parser.parse_args()

    logging.info(f'getting dataset from {args.data_path}')
    _dset = get_aa_dataset(args.data_path)

    # need to do manual sliding window as well

    logging.info(f'swapping the order of text and id')
    dset = []
    use_manual_sliding_window = not args.test_set
    if use_manual_sliding_window:
        logging.info('using manual sliding window')
    for dp in _dset:
        text = dp[1]
        lbl = int(dp[0])
        if len(text) > 512*5 and use_manual_sliding_window:
            # manual sliding windows
            for i in range(0, len(text), 512*5):
                dset.append([normalize_string(text[i:i+512*5]), lbl])
        else:
            dset.append([normalize_string(text[:512*6]), lbl])

    output_file = f'{args.data_path}.tsv'
    logging.info(f'writing the new tsv file to {output_file}')
    with open(output_file, 'w') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        writer.writerows(dset)
    logging.info('done!')
