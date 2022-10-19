import unittest
import os
import logging
from valla.dsets import loaders

logging.basicConfig(level=logging.INFO)

class TestDataset(unittest.TestCase):

    def setUp(self) -> None:
        pth = os.getenv('test_this_dset')
        dset_name = os.getenv('dset_name')
        dset_type = os.getenv('dset_type')

        self.dset_name = dset_name
        self.dset_type = dset_type
        self.check_unique = False

        self.val_av_unique = None
        self.test_av_unique = None

        if dset_type in ['', 'iid', 'av']:
            postfix = '_'
            if dset_type == 'av':
                self.check_unique = True
        elif dset_type == 'cross_topic':
            postfix = '_cross_topic_'
        elif dset_type == 'cross_genre':
            postfix = '_cross_genre_'
        else:
            self.assertEqual(0, 1, 'the dset type was not set properly')
            postfix = ''

        tmp = os.path.join(pth, dset_name+'*')
        logging.info(f'testing unique: {self.check_unique}')
        logging.info(f'getting datasets at: {tmp}')

        # get training set
        self.train_set = loaders.get_aa_dataset(os.path.join(pth, f'{dset_name}_train.csv'))
        # get aa sets
        self.val_aa_set = loaders.get_aa_dataset(os.path.join(pth, f'{dset_name}_AA{postfix}val.csv'))
        self.test_aa_set = loaders.get_aa_dataset(os.path.join(pth, f'{dset_name}_AA{postfix}test.csv'))
        # get av sets
        self.val_av_set = loaders.get_av_dataset(os.path.join(pth, f'{dset_name}_AV{postfix}val.csv'))
        self.test_av_set = loaders.get_av_dataset(os.path.join(pth, f'{dset_name}_AV{postfix}test.csv'))

        if self.check_unique:
            self.val_av_unique = loaders.get_av_dataset(os.path.join(pth, f'{dset_name}_AV_unique_val.csv'))
            self.test_av_unique = loaders.get_av_dataset(os.path.join(pth, f'{dset_name}_AV_unique_test.csv'))

    def test_print_length(self):
        logging.info(f'{len(self.train_set)} train samples')
        logging.info(f'{len(self.val_aa_set)} val samples')
        logging.info(f'{len(self.test_aa_set)} test samples')
        logging.info(f'{len(self.val_av_set)} val av samples')
        logging.info(f'{len(self.test_av_set)} test av samples')

    def check_av_lbls(self, av_dset, name):
        self.assertIn(0, av_dset, f'zero was not a label in the {name} av dataset')
        self.assertIn(1, av_dset, f'one was not a label in the {name} av dataset')
        self.assertEqual(2, len(av_dset), f'{name} does not have two components')

    def test_labels(self):
        # we need to check if the author numbers are correct but they don't have to be the same in each dataset
        train_lbls = set([auth_num for auth_num, _ in self.train_set])
        val_aa_lbls = set([auth_num for auth_num, _ in self.val_aa_set])
        test_aa_lbls = set([auth_num for auth_num, _ in self.test_aa_set])
        val_av_lbls = set([l for l, _, _ in self.val_av_set])
        test_av_lbls = set([l for l, _, _ in self.test_av_set])

        # make sure av labels are correct
        for av_dset, name in zip([val_av_lbls, test_av_lbls], ['val', 'test']):
            self.check_av_lbls(av_dset, name)

        # make sure test labels are correct
        correct_labels = set([x for x in range(len(train_lbls))])
        self.assertSetEqual(train_lbls, correct_labels, 'training set labels do not increment from 0 to len(train_set)')

        # make sure validation and test labels are in the training set labels
        for lbl in val_aa_lbls:
            self.assertIn(lbl, train_lbls, f'{lbl} in the val set was not found in the test set')
        for lbl in test_aa_lbls:
            self.assertIn(lbl, train_lbls, f'{lbl} in the test set was not found in the test set')

        logging.info('checking for equal auths in train/val/test sets - this is only true for some datasets so not asserting')
        logging.info(f'train and val check: {train_lbls == val_aa_lbls}')
        logging.info(f'train and test check: {train_lbls == test_aa_lbls}')
        logging.info(f'val and test check: {test_aa_lbls == val_aa_lbls}')

        if self.check_unique:
            logging.info(f'checking unique sets for proper labels')
            val_uav_lbls = set([l for l, _, _ in self.val_av_unique])
            test_uav_lbls = set([l for l, _, _ in self.test_av_unique])
            self.check_av_lbls(val_uav_lbls, 'unique_av_val')
            self.check_av_lbls(test_uav_lbls, 'unique_av_test')

    def test_non_overlapping(self):
        val_texts = [text for _, text in self.val_aa_set]
        test_texts = [text for _, text in self.test_aa_set]
        train_texts = [text for _, text in self.train_set]

        clash1, clash2 = [], []

        for idx, test_text in enumerate(test_texts):
            if test_text in val_texts:
                clash1.append(idx)
            if test_text in train_texts:
                clash2.append(idx)

        self.assertEqual(0, len(clash2), f'{len(clash2)} test texts were found in the train set')
        self.assertEqual(0, len(clash1), f'{len(clash1)} test texts were found in the validation set')

        for test_text in test_texts:
            self.assertNotIn(test_text, val_texts, 'a test text was found in the validation set')
            self.assertNotIn(test_text, train_texts, 'a test text was found in the train set')

        for train_text in train_texts:
            self.assertNotIn(train_text, test_texts, 'a train text was found in the test set')
            self.assertNotIn(train_text, val_texts, 'a train text was found in the validation set')

    def test_aa_av_splits(self):
        if (self.dset_name == 'guardian' and self.dset_type != 'iid') or self.dset_type == 'av':
            # skip the tests - this dset requries some special handling because it is so small
            return
        for aa_set, av_set, split in [(self.val_aa_set, self.val_av_set, 'val'),
                                      (self.test_aa_set, self.test_av_set, 'test')]:
            all_texts = [text for _, text in aa_set]
            set_av_texts = []
            for samediff, text1, text2 in av_set:
                set_av_texts.append(text1)
                set_av_texts.append(text2)
            set_av_texts = set(set_av_texts)
            self.assertSetEqual(set_av_texts, set(all_texts),
                                f'the av and aa {split} splits contain different texts')

    def tearDown(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()
