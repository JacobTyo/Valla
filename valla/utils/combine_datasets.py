import argparse
from valla.dsets.loaders import get_aa_dataset
from valla.utils.dataset_utils import write_aa_dataset,  list_dset_to_dict
import argparse
import logging

if __name__ == "__main__":
    # use this to normalize a dataset w.r.t. this tweet preprocessing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset1', type=str)
    parser.add_argument('--dataset2', type=str)
    parser.add_argument('--dataset_save_path', type=str)
    parser.add_argument('--remove_auths1', type=int, default=50)
    args = parser.parse_args()

    dset1 = list_dset_to_dict(get_aa_dataset(args.dataset1))
    dset2 = list_dset_to_dict(get_aa_dataset(args.dataset2))

    # remove used authors
    auth_id_counter = args.remove_auths1
    for auth in range(auth_id_counter):
        dset1.pop(auth)

    for auth, texts in dset1.items():
        dset2[auth_id_counter] = texts
        auth_id_counter += 1

    # now save combined dataset
    write_aa_dataset(dset2, f'{args.dataset_save_path}')
