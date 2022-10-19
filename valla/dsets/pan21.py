# Data access: https://pan.webis.de/data.html#pan20-authorship-verification-large
#  Specific files used (both train and test sets):
#    - PAN20-Authorship-Verification
#    - PAN20-Authorship-Verification (Large)
#    - PAN21-Authorship-Verification
from valla.dsets.pan20 import get_jsonl_file, normalize_pairs
from valla.utils.dataset_utils import write_av_dataset
import logging
import os
import argparse
import numpy as np
import random

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser(description='Get args for test dataset for pan21')

    parser.add_argument('--test_path', type=str)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    logging.info(f'getting the test dataset file form {args.test_path}')
    av_test_pairs = get_jsonl_file(os.path.join(args.test_path, 'raw', 'pan21-authorship-verification-test.jsonl'))
    av_test_set = normalize_pairs(av_test_pairs)

    # now write this dataset to the right path
    save_path = os.path.join(args.test_path, 'processed', 'pan21_AV_test.csv')
    write_av_dataset(av_test_set, save_path)
