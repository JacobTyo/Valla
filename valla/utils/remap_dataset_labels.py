from valla.dsets.loaders import get_aa_dataset, aa_as_pandas
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    # get each text in the dataset and do one same author and one different author. . . at least??
    parser = argparse.ArgumentParser(description='make sure labels are 0-len(labels.unique)')

    parser.add_argument('--file_path', type=str, nargs='+', default=[],
                        help='if a list of files is given, it will build the id map with the first file and apply to '
                             'the others.')

    args = parser.parse_args()

    old_to_new = {}
    first_file = True

    for old_file_path in args.file_path:
        file_name = old_file_path.split('/')[-1]
        logging.info(f'getting {file_name}')
        dset = aa_as_pandas(get_aa_dataset(old_file_path))
        if first_file:
            logging.info('building the old_to_new label map')
            counter = 0
            for lbl in dset.labels:
                if lbl not in old_to_new:
                    old_to_new[lbl] = counter
                    counter += 1
            first_file = False
        else:
            logging.info('reusing the old_to_new label map')

        dset = dset.rename(columns={'labels': 'old_labels'})

        logging.info('creating new labels')
        dset['labels'] = dset.old_labels.apply(lambda x: old_to_new[x])

        logging.info('removing old labels')
        dset = dset.drop(columns=['old_labels'])

        new_path = old_file_path + '.lbl_fix'
        logging.info(f'saving updated file to {new_path}')
        dset.to_csv(path_or_buf=new_path, columns=['labels', 'text'], header=True, index=False)
        logging.info(f'finished with {file_name}')
