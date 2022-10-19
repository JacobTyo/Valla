import argparse
from valla.dsets.loaders import get_aa_dataset
from valla.utils.dataset_utils import write_aa_dataset,  list_dset_to_dict
import argparse
import logging

if __name__ == "__main__":
    # use this to normalize a dataset w.r.t. this tweet preprocessing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--num_auths', type=int, default=1000)
    parser.add_argument('--inv', action='store_true')
    args = parser.parse_args()

    dset = get_aa_dataset(args.dataset_path)
    if args.inv:
        shrunk_dset = [[a, t] for a, t in dset if int(a) >= args.num_auths]
    else:
        shrunk_dset = [[a, t] for a, t in dset if int(a) < args.num_auths]
    # now save smaller set
    save_name = f'{args.dataset_path}.{args.num_auths}' if not args.inv else f'{args.dataset_path}.{args.num_auths}.inv'
    write_aa_dataset(list_dset_to_dict(shrunk_dset), save_name)
