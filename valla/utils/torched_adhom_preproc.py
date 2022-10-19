import argparse
import functools

import spacy
import textacy.preprocessing as tp
import csv
import time
import json
import sys
from multiprocessing import Pool
import logging

logging.basicConfig(level=logging.DEBUG)
csv.field_size_limit(sys.maxsize)
tokenizer = spacy.load('en_core_web_lg')


def preprocess_test_set(doc0, doc1, samediff):
    doc0 = preprocess(doc0)
    doc1 = preprocess(doc1)
    return samediff, doc0, doc1


def preprocess(doc, auth_id=None, count=False):
    # pre-process data
    doc = tp.normalize.unicode(doc)
    doc = tp.normalize.whitespace(doc)
    doc = tp.normalize.quotation_marks(doc)

    # apply spaCy to tokenize doc
    doc = tokenizer(doc)

    # build new sentences for pre-processed doc
    doc_new = []
    chars = []
    for sent in doc.sents:
        sent_new = ''
        this_sent_chars = []
        for token in sent:
            token = token.text
            token = token.replace('\n', '')
            token = token.replace('\t', '')
            token = token.strip()
            this_tok_chars = []
            for char in token:
                this_tok_chars.append(char)
            this_sent_chars.append(this_tok_chars)
            sent_new += token + ' '
        chars.append(this_sent_chars)
        doc_new.append(sent_new[:-1])

    return_dict = {
        'chars': chars,
        'tokens': doc_new
    }

    if auth_id is not None:
        if count:
            dict_chr_counts, dict_token_counts = count_tokens_and_characters(doc_new)
            return return_dict, auth_id, dict_chr_counts, dict_token_counts
        return return_dict, auth_id
    else:
        if count:
            dict_chr_counts, dict_token_counts = count_tokens_and_characters(doc_new)
            return return_dict, dict_chr_counts, dict_token_counts
        return return_dict

def count_tokens_and_characters(doc):
    dict_chr_counts, dict_token_counts = {}, {}
    for sent in doc:
        tokens = sent.split()
        for token in tokens:
            for chr in token:
                if chr not in dict_chr_counts:
                    dict_chr_counts[chr] = 0
                dict_chr_counts[chr] += 1
            if token not in dict_token_counts:
                dict_token_counts[token] = 0
            dict_token_counts[token] += 1
    return dict_chr_counts, dict_token_counts


def preprocess_and_count(doc):
    doc = preprocess(doc)
    return count_tokens_and_characters(doc)


class TokenizeAdHominem:
    def __init__(self, train_path, val_path, test_path):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        self.char_counts = {}
        self.tok_counts = {}

        self.tokenized_train = []
        self.tokenized_val = []
        self.tokenized_test = []

    def update_self(self, chr_counts, tok_counts):
        for k, v in chr_counts.items():
            if k not in self.char_counts:
                self.char_counts[k] = 0
            self.char_counts[k] += v

        for k, v in tok_counts.items():
            if k not in self.tok_counts:
                self.tok_counts[k] = 0
            self.tok_counts[k] += v

    def update_tokenized(self, tokenized, auth_id, chr_counts, tok_counts):
        self.tokenized_train.append([auth_id, tokenized])
        self.update_self(chr_counts, tok_counts)

    def update_tokenized_test(self, samediff, tokenized0, tokenized1, dset_type='test'):
        if dset_type == 'test':
            self.tokenized_test.append([samediff, tokenized0, tokenized1])
        elif dset_type == 'val':
            self.tokenized_val.append([samediff, tokenized0, tokenized1])
        else:
            raise ValueError('the dataset type was not recognized.')

    def tokenize_dataset(self, num_workers=10, test_only=False):
        if not test_only:
            async_results = {}
            with open(self.train_path, 'r') as train_file:
                reader = csv.reader(train_file)
                with Pool(processes=num_workers) as pool:

                    logging.info(f'launching the preprocessing to tokenize the train dataset, using {num_workers} workers')
                    for i, row in enumerate(reader):
                        if i == 0:
                            # skip header
                            continue
                        idx = i - 1
                        async_results[idx] = pool.apply_async(preprocess, (row[1][:100000], row[0], True))

                    logging.info('finished launching, now awaiting the processing')
                    done = False
                    start_time = time.time()
                    last_check = time.time()
                    loops = 0
                    num_removed = 0
                    while not done:
                        remove_idxs = []
                        loops += 1
                        for idx, result in async_results.items():
                            if result.ready():
                                res = result.get()
                                # do the things with this data. . .
                                self.update_tokenized(*res)
                                remove_idxs.append(idx)
                        num_removed += len(remove_idxs)
                        for idx in remove_idxs:
                            del async_results[idx]
                        if len(async_results.keys()) < 1:
                            done = True
                        if time.time() - last_check > 30:
                            elapsed = time.time() - start_time
                            res_left = len(list(async_results.keys()))
                            logging.info(
                                f'Tokenizing: {res_left} results remaining in queue, {elapsed:.2f}, {num_removed} removed, {loops} loops ran.')
                            logging.info(f'Tokenizing: approximately {(res_left / (num_removed / 30))/60} minutes remaining')
                            last_check = time.time()
                            num_removed = 0

                logging.info('all training data tokenized')
                # now save the tokenized dataset
                train_tokenized_path = self.train_path + '.adhom.tokenized'
                logging.info(f'saving the tokenized training dataset to: {train_tokenized_path}')
                with open(train_tokenized_path, 'w') as f:
                    writer = csv.writer(f)
                    for sample in self.tokenized_train:
                        writer.writerow([sample[0], json.dumps(sample[1])])

                # now save the training counts
                chr_count_filepath = self.train_path + '.adhom.chr_count'
                tok_count_filepath = self.train_path + '.adhom.tok_count'
                logging.info(f'saving char counts to: {chr_count_filepath}')
                logging.info(f'saving token counts to: {tok_count_filepath}')
                with open(chr_count_filepath, 'w') as chr_count_file:
                    json.dump(self.char_counts, chr_count_file, sort_keys=True, indent=2)

                with open(tok_count_filepath, 'w') as tok_count_file:
                    json.dump(self.tok_counts, tok_count_file, sort_keys=True, indent=2)

        for set_type, dset_path in zip(['val', 'test'], [self.val_path, self.test_path]):
            if dset_path is None:
                continue
            logging.info(f'now tokenizing the {set_type} set')
            async_results = {}
            with open(dset_path, 'r') as test_file:
                reader = csv.reader(test_file)
                with Pool(processes=num_workers) as pool:

                    logging.info(f'launching the preprocessing to tokenize the {set_type} dataset, using {num_workers} workers')
                    for i, row in enumerate(reader):
                        if i == 0:
                            # skip header
                            continue
                        idx = i - 1
                        async_results[idx] = pool.apply_async(preprocess_test_set, (row[1][:100000], row[2][:100000], row[0]))

                    logging.info('finished launching, now awaiting the processing')
                    done = False
                    start_time = time.time()
                    last_check = time.time()
                    loops = 0
                    num_removed = 0
                    while not done:
                        remove_idxs = []
                        loops += 1
                        for idx, result in async_results.items():
                            if result.ready():
                                res = result.get()
                                # do the things with this data. . .
                                self.update_tokenized_test(*res, dset_type=set_type)
                                remove_idxs.append(idx)
                        num_removed += len(remove_idxs)
                        for idx in remove_idxs:
                            del async_results[idx]
                        if len(async_results.keys()) < 1:
                            done = True
                        if time.time() - last_check > 30:
                            elapsed = time.time() - start_time
                            res_left = len(list(async_results.keys()))
                            logging.info(
                                f'Tokenizing {set_type}: {res_left} results remaining in queue, {elapsed:.2f}, {num_removed} removed, {loops} loops ran.')
                            logging.info(f'Tokenizing: approximately {(res_left / (num_removed / 30))/60} minutes remaining')
                            last_check = time.time()
                            num_removed = 0

                    logging.info(f'all {set_type} data tokenized')
                    # now save the tokenized dataset
                    test_tokenized_path = dset_path + '.adhom.tokenized'
                    logging.info(f'saving the tokenized {set_type} dataset to: {test_tokenized_path}')
                    with open(test_tokenized_path, 'w') as f:
                        writer = csv.writer(f)
                        save_me = self.tokenized_test if set_type == 'test' else self.tokenized_val
                        for sample in save_me:
                            writer.writerow([sample[0], json.dumps(sample[1]), json.dumps(sample[2])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--val_path', type=str, default=None)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--test_only', action='store_true')
    args = parser.parse_args()

    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
    num_workers = args.num_workers
    test_only = args.test_only

    adhom_tokenizer = TokenizeAdHominem(train_path, val_path, test_path)
    logging.info('starting the tokenizing and counting')
    adhom_tokenizer.tokenize_dataset(num_workers, test_only)
    logging.info('finished')


if __name__ == '__main__':
    main()
