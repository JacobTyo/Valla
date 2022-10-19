# -*- coding: utf-8 -*-
import os
import re
import pickle
import argparse
import logging
import sys
from bs4 import BeautifulSoup
import spacy
import textacy.preprocessing as tp
import numpy as np
from tqdm import tqdm
import fasttext
import pandas as pd
from sklearn.utils import shuffle
import random
import csv
from multiprocessing import Pool
import time
import copy
from valla.dsets.loaders import get_av_as_pandas
from valla.utils.dataset_utils import lim_av_size

logging.basicConfig(level=logging.DEBUG)
csv.field_size_limit(sys.maxsize)
TOKENIZER = spacy.load('en_core_web_lg')


def load_glove_model(fp):
    print("Loading Glove Model")
    glove_model = {}
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            try:
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
            except:
                print(f'dropping the embedding for: {word}')
                print(f'some context: {split_line[0:5]}')

    print(f"{len(glove_model)} words loaded!")
    return glove_model


def remove_dollaz(txt):
    if isinstance(txt, pd.Series):
        txt = txt.str.encode('utf-8').str.decode('utf-8')
        txt = txt.str.replace(r"(\$\$\$)", "&&&")
        txt = txt.str.slice(stop=489_999)
    else:
        txt = txt.encode('utf-8').decode('utf-8')
        txt = re.sub(r"(\$\$\$)", "&&&", txt)
        txt.replace('$$$', '&&&')
        # limit for some of the nltk tools
        if len(txt) > 499_000:
            txt = txt[:498_999]
    return txt


def add_special_tokens_doc_multiproc(doc, T_w):
    # add <SOS>
    N_w = []
    for i, sent in enumerate(doc):
        tokens = sent.split()
        doc[i] = ['<SOS>'] + tokens
        N_w.append(len(doc[i]))

    # add <EOS> or <ELB> or <SLB>
    doc_new = []
    for i, sent in enumerate(doc):
        # short sentence
        if N_w[i] <= T_w - 1:
            tokens = sent + ['<EOS>']
            doc_new.append(' '.join(tokens))
        # long sentence
        else:
            while len(sent) > 1:
                if len(sent) <= T_w - 1:
                    tokens = sent[:T_w - 1] + ['<EOS>']
                    doc_new.append(' '.join(tokens))
                else:
                    tokens = sent[:T_w - 1] + ['<ELB>']
                    doc_new.append(' '.join(tokens))
                sent = ['<SLB>'] + sent[T_w - 1:]

    return doc_new


def count_tokens_and_characters_multiproc(doc):
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


def preprocess_multiproc(doc): #, tokenizer):
    # pre-process data
    doc = tp.normalize.unicode(doc)
    doc = tp.normalize.whitespace(doc)
    doc = tp.normalize.quotation_marks(doc)

    # apply spaCy to tokenize doc
    doc = TOKENIZER(doc)

    # build new sentences for pre-processed doc
    doc_new = []
    for sent in doc.sents:
        sent_new = ''
        for token in sent:
            token = token.text
            token = token.replace('\n', '')
            token = token.replace('\t', '')
            token = token.strip()
            sent_new += token + ' '
        doc_new.append(sent_new[:-1])
    return doc_new


def extract_docs_work(review, label, is_test_datapoint, tokenizer=None, T_w=None):
    tmp = review.split('$$$')
    if len(tmp) > 2:
        print(f'there were {len(tmp)} texts after splitting. . . just getting first and last, {len(tmp[0])}, {len(tmp[-1])}')
        doc1, doc2 = tmp[0], tmp[-1]
    else:
        doc1, doc2 = tmp

    # if random.uniform(0, 1) < 0.5:
    #     doc_1 = BeautifulSoup(temp[0], 'html.parser').get_text().encode('utf-8').decode('utf-8')
    #     doc_2 = BeautifulSoup(temp[1], 'html.parser').get_text().encode('utf-8').decode('utf-8')
    # else:
    #     doc_2 = BeautifulSoup(temp[0], 'html.parser').get_text().encode('utf-8').decode('utf-8')
    #     doc_1 = BeautifulSoup(temp[1], 'html.parser').get_text().encode('utf-8').decode('utf-8')

    # preprocessing and tokenizing
    doc_1 = preprocess_multiproc(doc1)  #.encode('utf-8').decode('utf-8'))  # , tokenizer)
    doc_2 = preprocess_multiproc(doc2)  #.encode('utf-8').decode('utf-8'))  # , tokenizer)

    char_counts1, token_counts1 = None, None
    char_counts2, token_counts2 = None, None

    if not is_test_datapoint:
        # count tokens/characters in train set
        char_counts1, token_counts1 = count_tokens_and_characters_multiproc(doc_1)
        char_counts2, token_counts2 = count_tokens_and_characters_multiproc(doc_2)

    # add special tokens
    doc_1 = add_special_tokens_doc_multiproc(doc_1, T_w)
    doc_2 = add_special_tokens_doc_multiproc(doc_2, T_w)

    return doc_1, doc_2, label, is_test_datapoint, char_counts1, token_counts1, char_counts2, token_counts2


class Corpus(object):

    """
        Class for data preprocessing (8000 Amazon review pairs)
    """
    def __init__(self, test_split=0.2, T_w=20, D_w=300, vocab_size_token=15000, vocab_size_chr=125, dataset='amazon',
                 train_path=None, test_path=None, embeddings='fasttext'):

        # define Spacy tokenizer
        self.tokenizer = spacy.load('en_core_web_lg')

        # load raw data into memory - read in the training data into a pandas dataframe
        # I think we want two columns
        #   reviews: both text snippets, separted by $$$
        #   sentiment: label

        logging.debug('importing the training dataset')

        df = lim_av_size(get_av_as_pandas(train_path))
        df['is_test_datapoint'] = 0

        logging.debug('removing training dollaz')

        df['text0'] = remove_dollaz(df['text0'])
        df['text1'] = remove_dollaz(df['text1'])

        logging.debug('importing the test dataset')

        df_test = lim_av_size(get_av_as_pandas(test_path))
        df_test['is_test_datapoint'] = 1

        logging.debug('removing testing dollaz')

        df_test['text0'] = remove_dollaz(df_test['text0'])
        df_test['text1'] = remove_dollaz(df_test['text1'])

        logging.debug('concatenating the datasets')

        self.data_panda = pd.concat([df, df_test], ignore_index=True)

        # save on memory
        del df, df_test

        logging.debug('finalizing the dataset')

        self.data_panda = self.data_panda.rename(columns={'same/diff': 'sentiment'})
        self.data_panda['text0'] = self.data_panda['text0'] + '$$$' + self.data_panda['text1']

        logging.debug('dropping cols')
        self.data_panda = self.data_panda.rename(columns={'text0': 'review'})
        self.data_panda = self.data_panda.drop(columns=['text1'])

        logging.debug(f'we have these columns: {list(self.data_panda.columns)}')

        # load pre-trained fastText word embedding model
        if embeddings == 'fasttext':
            self.WE_dic = fasttext.load_model(os.path.join('data', 'cc.en.300.bin'))
        elif embeddings == 'glove':
            self.WE_dic = load_glove_model('glove.840B.300d.txt')
        else:
            assert False, f'{embeddings} is not a supported embedding'

        # dimension of word embeddings
        self.D_w = D_w
        # maximum words per sentence
        self.T_w = T_w

        # split size of test set
        self.test_split = test_split

        # train set
        self.docs_L_tr = []
        self.docs_R_tr = []
        self.labels_tr = []

        # test set
        self.docs_L_te = []
        self.docs_R_te = []
        self.labels_te = []

        # vocabulary sizes
        self.vocab_size_token = vocab_size_token
        self.vocab_size_chr = vocab_size_chr

        # token/word-based vocabulary
        self.V_w = {'<ZP>': 0,  # zero-padding
                    '<UNK>': 1,  # unknown token
                    '<SOS>': 2,  # start of sentence
                    '<EOS>': 3,  # end of sentence
                    '<SLB>': 4,  # start with line-break
                    '<ELB>': 5,  # end with line-break
                    }
        # character vocabulary
        self.V_c = {'<ZP>': 0,  # zero-padding character
                    '<UNK>': 1,  # "unknown"-character
                    }

        # dictionary with token/character counts
        self.dict_token_counts = {}
        self.dict_chr_counts = {}

        # unique list of most frequent tokens/characters
        self.list_tokens = None
        self.list_characters = None

        # word embedding matrix
        self.E_w = None


    def update_self(self, doc_1, doc_2, label, is_test_datapoint, char_counts1, token_counts1, char_counts2, token_counts2):
        if not is_test_datapoint:
            self.update_count_tokens_and_characters(char_counts1, token_counts1, char_counts2, token_counts2)

        if is_test_datapoint:
            # ad doc-pair to test set
            self.docs_L_te.append(doc_1)
            self.docs_R_te.append(doc_2)
            self.labels_te.append(label)

        else:
            # add doc-pair to train set
            self.docs_L_tr.append(doc_1)
            self.docs_R_tr.append(doc_2)
            self.labels_tr.append(label)


    def update_count_tokens_and_characters(self, char_counts1, token_counts1, char_counts2, token_counts2):
        # the inputs are dicts and need added to the self dict object
        for chars, tokens in [(char_counts1, token_counts1), (char_counts2, token_counts2)]:

            for k, v in chars.items():
                if k not in self.dict_chr_counts:
                    self.dict_chr_counts[k] = v
                else:
                    self.dict_chr_counts[k] += v

            for k, v in tokens.items():
                if k not in self.dict_token_counts:
                    self.dict_token_counts[k] = v
                else:
                    self.dict_token_counts[k] += v

    # extract docs
    def extract_docs(self, num_workers=10):

        with Pool(processes=num_workers) as pool:

            async_results = {}

            for idx in tqdm(range(self.data_panda.review.shape[0]), desc='preprocess docs'):

                async_results[idx] = pool.apply_async(extract_docs_work, (copy.deepcopy(self.data_panda.review[idx]),
                                                                          copy.deepcopy(self.data_panda.sentiment[idx]),
                                                                          copy.deepcopy(self.data_panda.is_test_datapoint[idx]),
                                                                          None, #self.tokenizer,
                                                                          self.T_w))

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
                        self.update_self(*res)
                        remove_idxs.append(idx)
                num_removed += len(remove_idxs)
                for idx in remove_idxs:
                    del async_results[idx]
                if len(async_results.keys()) < 1:
                    done = True
                if time.time() - last_check > 30:
                    elapsed = time.time() - start_time
                    print(f'{len(list(async_results.keys()))} results remaining in queue, {elapsed:.2f}, {num_removed} removed, {loops} loops ran.', end='\r')
                    last_check = time.time()
                    num_removed = 0

        print('done!')

        self.docs_L_tr, self.docs_R_tr, self.labels_tr = shuffle(self.docs_L_tr, self.docs_R_tr, self.labels_tr)
        self.docs_L_te, self.docs_R_te, self.labels_te = shuffle(self.docs_L_te, self.docs_R_te, self.labels_te)

    # pre-process single document
    def preprocess_doc(self, doc):

        # pre-process data
        doc = tp.normalize.unicode(doc)
        doc = tp.normalize.whitespace(doc)
        doc = tp.normalize.quotation_marks(doc)

        # apply spaCy to tokenize doc
        doc = self.tokenizer(doc)

        # build new sentences for pre-processed doc
        doc_new = []
        for sent in doc.sents:
            sent_new = ''
            for token in sent:
                token = token.text
                token = token.replace('\n', '')
                token = token.replace('\t', '')
                token = token.strip()
                sent_new += token + ' '
            doc_new.append(sent_new[:-1])
        return doc_new

    # function for single document
    def add_special_tokens_doc(self, doc):

        # add <SOS>
        N_w = []
        for i, sent in enumerate(doc):
            tokens = sent.split()
            doc[i] = ['<SOS>'] + tokens
            N_w.append(len(doc[i]))

        # add <EOS> or <ELB> or <SLB>
        doc_new = []
        for i, sent in enumerate(doc):
            # short sentence
            if N_w[i] <= self.T_w - 1:
                tokens = sent + ['<EOS>']
                doc_new.append(' '.join(tokens))
            # long sentence
            else:
                while len(sent) > 1:
                    if len(sent) <= self.T_w - 1:
                        tokens = sent[:self.T_w - 1] + ['<EOS>']
                        doc_new.append(' '.join(tokens))
                    else:
                        tokens = sent[:self.T_w - 1] + ['<ELB>']
                        doc_new.append(' '.join(tokens))
                    sent = ['<SLB>'] + sent[self.T_w - 1:]

        # this is a list of sentences, so given a single doc, return a list of processed sentences, only T_w chars per sentence?
        return doc_new

    def count_tokens_and_characters(self, doc):
        for sent in doc:
            tokens = sent.split()
            for token in tokens:
                for chr in token:
                    if chr not in self.dict_chr_counts:
                        self.dict_chr_counts[chr] = 0
                    self.dict_chr_counts[chr] += 1
                if token not in self.dict_token_counts:
                    self.dict_token_counts[token] = 0
                self.dict_token_counts[token] += 1

    # remove rare tokens and characters
    def remove_rare_tok_chr(self):

        # remove rare token types
        q = sorted(self.dict_token_counts.items(), key=lambda x: x[1], reverse=True)
        self.list_tokens = list(list(zip(*q))[0])[:self.vocab_size_token]

        # remove rare character types
        q = sorted(self.dict_chr_counts.items(), key=lambda x: x[1], reverse=True)
        self.list_characters = list(list(zip(*q))[0])[:self.vocab_size_chr]

    # make word- and character-based vocabularies
    def make_wrd_chr_vocabularies(self):

        # add tokens to vocabulary and assign an integer
        for token in self.list_tokens:
            self.V_w[token] = len(self.V_w)

        # word embedding matrix
        self.E_w = np.zeros(shape=(len(self.V_w), self.D_w), dtype='float32')
        r = np.sqrt(3.0 / self.D_w)
        for token in self.V_w.keys():
            idx = self.V_w[token]
            if token in ['<UNK>', '<SOS>', '<EOS>', '<SLB>', '<ELB>']:
                # initialize special tokens
                self.E_w[idx, :] = np.random.uniform(low=-r, high=r, size=(1, self.D_w))
            else:
                # initialize pre-trained tokens
                try:
                    self.E_w[idx, :] = self.WE_dic[token]
                except Exception:
                    # the <ZP> token wasn't found? replace it with something else?
                    self.E_w[idx, :] = self.WE_dic['SAM']

        for c in self.list_characters:
            self.V_c[c] = len(self.V_c)

    def extract_docs_single_thread(self):

        problems, train_probs, test_probs = 0, 0, 0
        l = self.data_panda.review.shape[0]
        logging.info(f'extracting {l} texts')

        for idx in tqdm(range(self.data_panda.review.shape[0]), desc='preprocess docs'):

            tmp = self.data_panda.review[idx].split('$$$')
            if len(tmp) > 2:
                print(
                    f'there were {len(tmp)} texts after splitting. . . just getting first and last, {len(tmp[0])}, {len(tmp[-1])}')
                doc1, doc2 = tmp[0], tmp[-1]
            else:
                doc1, doc2 = tmp


            # preprocessing and tokenizing
            doc_1 = self.preprocess_doc(doc1)
            doc_2 = self.preprocess_doc(doc2)

            if self.data_panda.is_test_datapoint[idx] != 1:
                # count tokens/characters in train set
                self.count_tokens_and_characters(doc_1)
                self.count_tokens_and_characters(doc_2)

            # add special tokens
            doc_1 = self.add_special_tokens_doc(doc_1)
            doc_2 = self.add_special_tokens_doc(doc_2)

            if self.data_panda.is_test_datapoint[idx] != 1:
                # add doc-pair to train set
                self.docs_L_tr.append(doc_1)
                self.docs_R_tr.append(doc_2)
                self.labels_tr.append(self.data_panda.sentiment[idx])

            else:
                # ad doc-pair to test set
                self.docs_L_te.append(doc_1)
                self.docs_R_te.append(doc_2)
                self.labels_te.append(self.data_panda.sentiment[idx])

        # shuffle
        self.docs_L_tr, self.docs_R_tr, self.labels_tr = shuffle(self.docs_L_tr, self.docs_R_tr, self.labels_tr)
        self.docs_L_te, self.docs_R_te, self.labels_te = shuffle(self.docs_L_te, self.docs_R_te, self.labels_te)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdHominem - preprocessing')

    # self, test_split = 0.2, T_w = 20, D_w = 300, vocab_size_token = 15000, vocab_size_chr = 125, dataset = 'amazon',
    # train_path = None, test_path = None, embeddings = 'fasttext'

    parser.add_argument('--dataset', default='amazon', type=str)  # character embedding dimension
    parser.add_argument('--train_path', default='/home/jtyo/Repos/AuthorshipAttribution/data/_gutenburg/train.csv', type=str)
    parser.add_argument('--test_path', default='/home/jtyo/Repos/AuthorshipAttribution/data/_gutenburg/test.csv', type=str)
    parser.add_argument('--embeddings', default='fasttext', type=str)
    parser.add_argument('--save_path', type=str, default='data')
    # '/home/jtyo/Repos/AuthorshipAttribution/data/_gutenburg/train_test_adhominem_glove.pkl'
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--vocab_size_token', type=int, default=5000)
    parser.add_argument('--vocab_size_char', type=int, default=300)
    parser.add_argument('--T_w', type=int, default=30)
    parser.add_argument('--D_w', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--sweep', action='store_true')

    args = parser.parse_args()
    hyper_parameters = vars(args)

    test_val = 'test' if 'test' in args.test_path else 'val'
    distshift = ''
    if 'cross_topic' in args.test_path:
        distshift = 'ct'
    elif 'cross_genre' in args.test_path:
        distshift = 'cg'
    elif 'unique' in args.test_path:
        distshift = 'ua'
    else:
        distshift = 'iid'

    # for hyperparameter sweeping
    if args.sweep:
        list_vocab_size_token = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 5000, 5000, 5000, 5000, 5000, 5000]
        list_vocab_size_chr = [150, 150, 150, 150, 150, 150, 150, 100, 200, 250, 500, 1000, 2000]
    else:
        list_vocab_size_token = [args.vocab_size_token]
        list_vocab_size_chr = [args.vocab_size_char]

    for vocab_size_token, vocab_size_chr in tqdm(zip(list_vocab_size_token, list_vocab_size_chr)):

        corpus = Corpus(dataset=args.dataset, train_path=args.train_path, test_path=args.test_path,
                        embeddings=args.embeddings, vocab_size_token=vocab_size_token,
                        vocab_size_chr=vocab_size_chr, T_w=args.T_w, D_w=args.D_w)
        if args.distributed:
            corpus.extract_docs(num_workers=args.num_workers)
        else:
            corpus.extract_docs_single_thread()
        # cache the state here I think

        corpus.remove_rare_tok_chr()
        corpus.make_wrd_chr_vocabularies()

        # make sure the dir exists and make if not
        os.makedirs(args.save_path, exist_ok=True)
        output_path = os.path.join(args.save_path, f'{args.dataset}_{test_val}_{distshift}_{args.embeddings}_'
                                                   f'{vocab_size_token}_'
                                                   f'{vocab_size_chr}_{args.T_w}_{args.D_w}.pkl')

        with open(output_path, 'wb') as f:
            pickle.dump((corpus.docs_L_tr, corpus.docs_R_tr, corpus.labels_tr,
                         corpus.docs_L_te, corpus.docs_R_te, corpus.labels_te,
                         corpus.V_w, corpus.E_w, corpus.V_c), f)
