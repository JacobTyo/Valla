import argparse
from valla.dsets.loaders import get_aa_dataset
from valla.utils.dataset_utils import write_aa_dataset,  list_dset_to_dict
import argparse
import logging

if __name__ == "__main__":
    # use this to normalize a dataset w.r.t. this tweet preprocessing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--num_texts', type=int, default=50)
    args = parser.parse_args()

    dset = list_dset_to_dict(get_aa_dataset(args.dataset_path))
    for auth in list(dset.keys()):
        dset[auth] = dset[auth][:args.num_texts]
    # now save smaller set
    write_aa_dataset(dset, f'{args.dataset_path}.lt{args.num_texts}')
