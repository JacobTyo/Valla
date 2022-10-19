"""
This is our pytorch implementation of "https://github.com/boenninghoff/AdHominem"
"""

import argparse

import torch
import torchtext.vocab
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
import torchtext.transforms as T
import os
import sys
import time
import csv
import json
import random
import logging
import wandb
import functools
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from spacy.vocab import Vocab
from spacy.language import Language
from spacy.lang.en import English
import textacy.preprocessing as tp
from valla.utils.eval_metrics import av_metrics, aa_metrics
from valla.dsets.loaders import get_av_dataset, get_aa_dataset
from valla.utils.dataset_utils import list_dset_to_dict
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logging.basicConfig(level=logging.DEBUG)
csv.field_size_limit(sys.maxsize)


def build_map_from_vocab(vocab, unknown_tok, padding_tok):
    # build a char-to-id map
    id_map = {
        unknown_tok: 0,
        padding_tok: 1
    }
    i = 1
    for thing in vocab:
        id_map[thing] = i
        i += 1
    return id_map


class AdHomDataset(Dataset):
    def __init__(self, char_vocab, tok_vocab, max_chars_per_word=15, max_words_per_sentence=20,
                 max_sentences_per_doc=50, dont_use_fasttext=False):
        self.UNK_TOKEN = '<UNK>'
        self.PAD_TOKEN = '<PAD>'
        self.FT_PAD_TOKEN = '-kathyVodka'
        self.FT_UNK_TOKEN = 'run25.3'
        self.max_chars_per_word = max_chars_per_word
        self.max_words_per_sentence = max_words_per_sentence
        self.max_sentences_per_doc = max_sentences_per_doc

        self.tok_vocab = tok_vocab
        self.char_vocab_size = len(char_vocab) + 2
        self.tok_vocab_size = len(tok_vocab) + 2

        self.dont_use_fasttext = dont_use_fasttext

        # do some weird stuff so we can get ONLY the tokenizer and sentence boundary detection from spacy without other
        # slow stuff
        self.tokenizer = Language(Vocab())
        good_tokenizer = English()
        self.tokenizer.tokenizer = good_tokenizer.tokenizer
        self.tokenizer.add_pipe('sentencizer')

        if dont_use_fasttext:
            self.tok_vocab_obj = vocab(tok_vocab, specials=[self.UNK_TOKEN, self.PAD_TOKEN])
            self.tok_vocab_obj.set_default_index(self.tok_vocab_obj[self.UNK_TOKEN])
            self.tok_transform = T.Sequential(
                T.VocabTransform(self.tok_vocab_obj),
                T.Truncate(self.max_words_per_sentence),
                T.ToTensor(self.tok_vocab_obj[self.PAD_TOKEN])
            )
            # TODO: !!!!!!!!!!!! if not using fast text need to set this for test set as well.
        else:
            self.fasttext = torchtext.vocab.FastText(language='en')

        self.char_vocab_obj = vocab(char_vocab, specials=[self.UNK_TOKEN, self.PAD_TOKEN])

        # set OOV chars to unknown token
        self.char_vocab_obj.set_default_index(self.char_vocab_obj[self.UNK_TOKEN])

        self.char_transform = T.Sequential(
            T.VocabTransform(self.char_vocab_obj),
            T.Truncate(self.max_chars_per_word),
            T.ToTensor(self.char_vocab_obj[self.PAD_TOKEN])
        )

        self.max_txt_len = (self.max_chars_per_word + 1) * self.max_words_per_sentence * self.max_sentences_per_doc

    def ids_to_tokens(self, chars, tokens):
        pad_char_id = self.char_vocab_obj([self.PAD_TOKEN])[0]
        c = torch.ones(size=(self.max_sentences_per_doc, self.max_words_per_sentence, self.max_chars_per_word)) * pad_char_id
        if self.dont_use_fasttext:
            t = self.tok_vocab_obj([self.PAD_TOKEN])[0]
            t = torch.ones(size=(self.max_sentences_per_doc, self.max_words_per_sentence)) * t
        else:
            t = self.fasttext.get_vecs_by_tokens(self.FT_PAD_TOKEN)
            t = t.repeat(self.max_sentences_per_doc, self.max_words_per_sentence, 1)
        ws_dropout_mask = torch.zeros(size=(self.max_sentences_per_doc, self.max_words_per_sentence)) - 5000
        sd_dropout_mask = torch.zeros(size=(self.max_sentences_per_doc,)) - 5000
        # TODO: the packing fails with empty sentence, noto sure what to do so just saying its len 1 for now
        ws_lens = torch.ones(size=(self.max_sentences_per_doc,))
        sd_len = 0
        for i, (sentence_c, sentence_t) in enumerate(zip(chars, tokens)):
            if i >= self.max_sentences_per_doc:
                break
            # chars to id
            transformed_c = self.char_transform(sentence_c)
            c_lim1 = min(self.max_words_per_sentence, transformed_c.shape[0])
            c_lim2 = min(self.max_chars_per_word, transformed_c.shape[1])
            c[i, :c_lim1, :c_lim2] = transformed_c[:c_lim1, :c_lim2]
            # tokens to fasttext vector or id
            if self.dont_use_fasttext:
                transformed_t = self.tok_transform(sentence_t.split())
                t_lim = min(self.max_words_per_sentence, transformed_t.shape[0])
                t[i, :t_lim] = transformed_t[:t_lim]
                t = t.type(torch.LongTensor)
            else:
                tokens = sentence_t.split()
                # use a limited vocabulary
                tokens = list(map(lambda x: x if x in self.tok_vocab else self.FT_UNK_TOKEN, tokens))
                try:
                    transformed_t = self.fasttext.get_vecs_by_tokens(tokens)
                except RuntimeError as e:
                    # not sure what is actually happening here = perhaps the len(tokens) is 0?
                    logging.error(f'error getting vectors from fasttext: {e}')
                    logging.error('just filling with an array of unknown tokens for now')
                    logging.error(f'len(tokens): {len(tokens)}')
                    logging.error(f'sentence_t: {sentence_t}')
                    transformed_t = self.fasttext.get_vecs_by_tokens(self.FT_UNK_TOKEN).repeat(self.max_words_per_sentence, 1)
                    pass
                t_lim = min(self.max_words_per_sentence, transformed_t.shape[0])
                t[i, :t_lim, :] = transformed_t[:t_lim, :]
            ws_dropout_mask[i, :t_lim] = torch.zeros(size=(t_lim,))
            sd_dropout_mask[i] = 0
            ws_lens[i] = t_lim
            sd_len += 1
        return c.type(torch.LongTensor), t, ws_dropout_mask, sd_dropout_mask, ws_lens, sd_len

    def get_char_transform(self):
        return self.char_transform

    def set_char_transform(self, char_transform):
        self.char_transform = char_transform

    def preprocess_doc(self, doc):
        doc = tp.normalize.unicode(doc)
        doc = doc.replace('\n', ' ')
        doc = doc.replace('\t', ' ')
        doc = tp.normalize.whitespace(doc)
        doc = tp.normalize.quotation_marks(doc)

        # apply spaCy to tokenize doc
        doc = self.tokenizer(doc)

        # build new sentences for pre-processed doc
        doc_new = []
        chars = []
        for sent in doc.sents:
            sent_new = ''
            this_sent_chars = []
            for token in sent:
                token = token.text
                token = token.strip()
                this_tok_chars = list(token)
                this_sent_chars.append(this_tok_chars)
                sent_new += token + ' '
            chars.append(this_sent_chars)
            doc_new.append(sent_new[:-1])

        return_dict = {
            'chars': chars,  # this should be (num_sents, num_words, num_chars)
            'tokens': doc_new  # this should be (numsentences, num_words)
        }
        return return_dict


class AADataset(AdHomDataset):
    def __init__(self, data_path, *args, **kwargs):
        super(AADataset, self).__init__(*args, **kwargs)

        # need to fix this, no longer just one json blob
        logging.info(f'loading dataset from {data_path}')
        self.data = list_dset_to_dict(get_aa_dataset(data_path))

        self.data_len = sum([len(x) for x in self.data.values()])

        # build a map to uniquely identify texts
        self.idx_to_txt_map = {}
        i = 0
        for auth, texts in self.data.items():
            text_loc = 0
            for _ in texts:
                self.idx_to_txt_map[i] = {
                    'auth_id': auth,
                    'text_id': text_loc
                }
                text_loc += 1
                i += 1

        self.author_list = list(self.data.keys())

        # self.fixed_data = {}

    def __len__(self):
        return self.data_len

    # def reset_sampled_data(self):
    #     self.fixed_data = {}

    def __getitem__(self, item):

        start_time = time.time()
        auth_id = self.idx_to_txt_map[item]['auth_id']
        txt_num = self.idx_to_txt_map[item]['text_id']
        # text0 is a list of sentences

        # if item in self.fixed_data:
        #
        #     label, raw_text, text1 = self.fixed_data[item]
        #     preprocessed = self.preprocess_doc(raw_text[:self.max_txt_len])
        #     tokens0, chars0 = preprocessed['tokens'], preprocessed['chars']
        #     preprocessed1 = self.preprocess_doc(text1[:self.max_txt_len])
        #     tokens1, chars1 = preprocessed1['tokens'], preprocessed1['chars']
        #
        # else:
        raw_text = self.data[auth_id][txt_num]

        preprocessed = self.preprocess_doc(raw_text[:self.max_txt_len])

        tokens0, chars0 = preprocessed['tokens'], preprocessed['chars']

        # return a randomly sampled tuple
        if random.random() < 0.5:
            # different author sample
            label = -1
            auth2 = random.choice(self.author_list)
            while auth2 == auth_id:
                auth2 = random.choice(self.author_list)
            text1 = random.choice(self.data[auth2])

        else:
            label = 1
            text1 = random.choice(self.data[auth_id])

        preprocessed1 = self.preprocess_doc(text1[:self.max_txt_len])
        tokens1, chars1 = preprocessed1['tokens'], preprocessed1['chars']

        # self.fixed_data[item] = label, raw_text, text1

        # now tokens is a list of lists of tokens
        #   and chars is a list of list of lists of characters
        #   need to transform those to ids/embeddings/whatever for the model
        chars0, tokens0, ws_do_mask0, sd_do_mask0, ws_lens0, sd_lens0 = self.ids_to_tokens(chars0, tokens0)
        chars1, tokens1, ws_do_mask1, sd_do_mask1, ws_lens1, sd_lens1 = self.ids_to_tokens(chars1, tokens1)

        processing_time = time.time() - start_time
        if processing_time > 10:
            # if this took longer than 10 seconds, print some stuff
            logging.warning(f'this sample took {processing_time:.0f} seconds to process, which is too long')
            logging.warning(f'auth_id: {auth_id}')
            logging.warning(f'text_num: {txt_num}')
            logging.warning(f'len(raw_text): {len(raw_text)}')

        return torch.tensor(label), tokens0, tokens1, chars0, chars1, ws_do_mask0, ws_do_mask1, sd_do_mask0, \
               sd_do_mask1, ws_lens0, ws_lens1, sd_lens0, sd_lens1


class AA_eval_dataset(AdHomDataset):
    def __init__(self, data_path, *args, **kwargs):
        super(AA_eval_dataset, self).__init__(*args, **kwargs)

        # need to fix this, no longer just one json blob
        logging.info(f'loading dataset from {data_path}')
        self.data = list_dset_to_dict(get_aa_dataset(data_path))

        self.data_len = sum([len(x) for x in self.data.values()])

        # build a map to uniquely identify texts
        self.idx_to_txt_map = {}
        i = 0
        for auth, texts in self.data.items():
            text_loc = 0
            for _ in texts:
                self.idx_to_txt_map[i] = {
                    'auth_id': auth,
                    'text_id': text_loc
                }
                text_loc += 1
                i += 1

        self.author_list = list(self.data.keys())

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        # this will only return the chars and whatnot of a single text, don't need the siamese setup here.
        auth_id = self.idx_to_txt_map[item]['auth_id']
        txt_num = self.idx_to_txt_map[item]['text_id']

        raw_text = self.data[auth_id][txt_num]

        preprocessed = self.preprocess_doc(raw_text[:self.max_txt_len])

        tokens0, chars0 = preprocessed['tokens'], preprocessed['chars']

        chars0, tokens0, ws_do_mask0, sd_do_mask0, ws_lens0, sd_lens0 = self.ids_to_tokens(chars0, tokens0)

        return torch.tensor(auth_id), tokens0, tokens0, chars0, chars0, ws_do_mask0, ws_do_mask0, sd_do_mask0, \
               sd_do_mask0, ws_lens0, ws_lens0, sd_lens0, sd_lens0


class AVDataset(AdHomDataset):

    def __init__(self, data_path, *args, **kwargs):
        # , char_vocab=None, tok_vocab=None, char_to_id=None, tok_to_id=None, **kwargs):
        super(AVDataset, self).__init__(*args, **kwargs)
        logging.info(f'loading dataset from {data_path}')
        self.data = get_av_dataset(data_path)

        self.data_len = len(self.data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        label, doc0, doc1 = self.data[item]
        doc0 = self.preprocess_doc(doc0[:self.max_txt_len])
        doc1 = self.preprocess_doc(doc1[:self.max_txt_len])
        chars0, tokens0 = doc0['chars'], doc0['tokens']
        chars1, tokens1 = doc1['chars'], doc1['tokens']

        label = label if label == 1 else -1
        chars0, tokens0, ws_do_mask0, sd_do_mask0, ws_lens0, sd_lens0 = self.ids_to_tokens(chars0, tokens0)
        chars1, tokens1, ws_do_mask1, sd_do_mask1, ws_lens1, sd_lens1 = self.ids_to_tokens(chars1, tokens1)

        return torch.tensor(label), tokens0, tokens1, chars0, chars1, ws_do_mask0, ws_do_mask1, sd_do_mask0, \
               sd_do_mask1, ws_lens0, ws_lens1, sd_lens0, sd_lens1


class AdHominem(torch.nn.Module):

    def __init__(self, params):
        super(AdHominem, self).__init__()

        # set an initial threshold, this will be adjusted in training if desired
        self.best_threshold = 0.5

        self.char_vocab_size = params.chr_vocab_size
        self.tok_vocab_size = params.tok_vocab_size
        self.cnn_stride = params.cnn_stride
        self.D_c = params.D_c
        self.D_r = params.D_r
        self.w = params.w
        self.D_w = params.D_w
        self.D_s = params.D_s
        self.D_d = params.D_d
        self.D_mlp = params.D_mlp
        self.cnn_dropout_prob = params.cnn_dropout_prob
        self.w2s_dropout_prob = params.w2s_dropout_prob
        self.w2s_att_dropout_prob = params.w2s_att_dropout_prob
        self.s2d_dropout_prob = params.s2d_dropout_prob
        self.s2d_att_dropout_prob = params.s2d_att_dropout_prob
        self.metric_dropout_prob = params.metric_dropout_prob

        self.batch_size = params.train_batch_size

        self.max_chars_per_word = params.max_chars_per_word
        self.max_words_per_sentence = params.max_words_per_sentence
        self.max_sentences_per_doc = params.max_sentences_per_doc

        self.dont_use_fasttext = params.dont_use_fasttext

        self.char_embedding = nn.Embedding(num_embeddings=self.char_vocab_size+2, embedding_dim=self.D_c)

        if self.dont_use_fasttext:
            self.word_embedding = nn.Embedding(num_embeddings=self.tok_vocab_size + 2, embedding_dim=self.D_w)
        else:
            assert self.D_w == 300, 'the word embedding dimension must be 300 when using pretrained fasttext'

        # cnn
        self.cnn_dropout = nn.Dropout(p=self.cnn_dropout_prob)
        self.cnn = nn.Conv1d(in_channels=self.D_c, out_channels=self.D_r, kernel_size=self.w,
                             padding=self.w - 1, stride=self.cnn_stride)
        # max over time pooling
        self.max_over_time = nn.MaxPool1d(kernel_size=self.max_chars_per_word + self.w - 1, stride=self.cnn_stride)

        # lstm word-to-sentence bidirectional
        self.word_to_sentence_lstm = nn.LSTM(input_size=self.D_w + self.D_r, hidden_size=self.D_s, bidirectional=True,
                                             dropout=self.w2s_dropout_prob, num_layers=1, batch_first=True)
        # manual dropout because pytorch dooesn't work on the first lstm layer
        self.word_to_sentence_lstm_do = nn.Dropout(p=self.w2s_dropout_prob)

        # attention word-to-sentence
        self.w2s_lin = nn.Linear(in_features=2*self.D_s, out_features=2*self.D_s)
        # self.w2s_lin_v = nn.Linear(in_features=2*self.D_s, out_features=1, bias=False)
        self.w2s_lin_v = nn.parameter.Parameter(nn.init.xavier_normal_(torch.zeros(size=(2*self.D_s, 1), requires_grad=True)))
        # self.word_to_sentence_att = nn.MultiheadAttention(embed_dim=2*self.D_s, num_heads=1,
        #                                                   dropout=self.w2s_att_dropout_prob, batch_first=True)
        self.w2s_att_do = nn.Dropout(p=self.w2s_att_dropout_prob)

        # lstm sentence-to-document bidirectional
        self.sentence_to_doc_lstm = nn.LSTM(input_size=2 * self.D_s, hidden_size=self.D_d,
                                            bidirectional=True, dropout=self.s2d_dropout_prob, num_layers=1,
                                            batch_first=True)

        # manual dropout because pytorch dooesn't work on the first lstm layer
        self.sentence_to_doc_lstm_do = nn.Dropout(p=self.s2d_dropout_prob)

        # attention sentence-to-document
        self.s2d_lin = nn.Linear(in_features=2 * self.D_d, out_features=2 * self.D_d)
        # self.s2d_lin_v = nn.Linear(in_features=2 * self.D_d, out_features=1, bias=False)
        self.s2d_lin_v = nn.parameter.Parameter(nn.init.xavier_normal_(torch.zeros(size=(2*self.D_d, 1), requires_grad=True)))
        # self.sentence_to_doc_att = nn.MultiheadAttention(embed_dim=2 * self.D_d, num_heads=1,
        #                                                  dropout=self.s2d_att_dropout_prob, batch_first=True)
        self.s2d_att_do = nn.Dropout(p=self.s2d_att_dropout_prob)

        # metric
        self.metric = nn.Linear(in_features=2 * self.D_d, out_features=self.D_mlp)
        self.metric_do = nn.Dropout(p=self.metric_dropout_prob)

        # testing
        self.max_over_time2 = nn.MaxPool1d(kernel_size=20, stride=1)
        self.max_over_time3 = nn.MaxPool1d(kernel_size=50, stride=1)

        # more testing
        self.test_layer1 = nn.Linear(in_features=15000, out_features=2056)
        self.test_layer2 = nn.Linear(in_features=2056, out_features=2056)
        self.test_layer3 = nn.Linear(in_features=2056, out_features=256)
        self.test_layer4 = nn.Linear(in_features=5000, out_features=2 * self.D_d)

        self.test_layer4_do = nn.Dropout(p=self.s2d_dropout_prob)

        # nonlinearity
        self.tanh = nn.Tanh()

    def w2s_att(self, _x, do_mask):
        batch_size = self.batch_size
        # make sure the input is shaped properly and store for later use
        _x = torch.reshape(_x,
                           shape=(batch_size, self.max_sentences_per_doc, self.max_words_per_sentence, 2 * self.D_s))
        # define the query and apply dropout
        x = torch.reshape(torch.clone(_x), shape=(batch_size*self.max_sentences_per_doc*self.max_words_per_sentence, 2*self.D_s))
        x = self.w2s_att_do(x)
        x = self.tanh(self.w2s_lin(x))
        # apply dropout again
        x = self.w2s_att_do(x)
        # multiply by values
        x = x@self.w2s_lin_v
        x = torch.reshape(x, shape=(batch_size, self.max_sentences_per_doc, self.max_words_per_sentence))
        # apply dropout mask
        x = x + do_mask
        x = torch.softmax(x, dim=2)
        x = torch.unsqueeze(x, dim=3)
        x = torch.tile(x, dims=(1, 1, 1, 2*self.D_s))
        return torch.sum(x * _x, dim=2)

    def s2d_att(self, _x, do_mask):
        batch_size = self.batch_size
        _x = torch.reshape(_x,
                           shape=(batch_size, self.max_sentences_per_doc, 2 * self.D_d))
        x = torch.reshape(torch.clone(_x), shape=(batch_size * self.max_sentences_per_doc, 2 * self.D_d))
        x = self.s2d_att_do(x)
        x = self.tanh(self.s2d_lin(x))
        x = self.s2d_att_do(x)
        x = x@self.s2d_lin_v
        x = torch.reshape(x, shape=(batch_size, self.max_sentences_per_doc))
        # apply dropout mask
        x = x + do_mask
        x = torch.softmax(x, dim=1)
        x = torch.unsqueeze(x, dim=2)
        x = torch.tile(x, dims=(1, 1, 2*self.D_d))
        return torch.sum(x * _x, dim=1)

    def char_cnn(self, chars):
        batch_size = chars.shape[0]
        num_sentences = chars.shape[1]
        num_words = chars.shape[2]
        num_chars = chars.shape[3]

        chars = self.char_embedding(chars)
        char_emb_size = chars.shape[-1]
        chars = torch.reshape(chars, [batch_size * num_sentences * num_words, num_chars, char_emb_size])
        chars = torch.permute(chars, (0, 2, 1))
        chars = self.cnn_dropout(chars)
        chars = self.max_over_time(self.tanh(self.cnn(chars)))
        chars = torch.squeeze(chars)
        char_cnn_out_dim = chars.shape[-1]
        chars = torch.reshape(chars, shape=(batch_size, num_sentences, num_words, char_cnn_out_dim))

        return chars

    def forward(self, tokens0, tokens1, char0, char1, ws_do_mask0, ws_do_mask1, sd_do_mask0, sd_do_mask1, ws_lens0,
                ws_lens1, sd_lens0, sd_lens1):

        batch_size = char0.shape[0]
        self.batch_size = batch_size
        num_sentences = char0.shape[1]
        num_words = char0.shape[2]

        # get the character representations from the char cnn
        char0 = self.char_cnn(char0)
        char1 = self.char_cnn(char1)

        if self.dont_use_fasttext:
            tokens0 = self.word_embedding(tokens0)
            tokens1 = self.word_embedding(tokens1)

        # combine the cnn representations with the word embeddings and reshape for w->s LSTM
        tokens0 = torch.reshape(torch.cat((char0, tokens0), dim=-1), shape=(batch_size * num_sentences, num_words, -1))
        tokens1 = torch.reshape(torch.cat((char1, tokens1), dim=-1), shape=(batch_size * num_sentences, num_words, -1))

        tokens0 = self.word_to_sentence_lstm_do(tokens0)
        tokens1 = self.word_to_sentence_lstm_do(tokens1)

        # pack the lstm input (i.e. mask the tokens that need masked)
        ws_lens0 = torch.reshape(ws_lens0, shape=(batch_size * num_sentences,))
        ws_lens1 = torch.reshape(ws_lens1, shape=(batch_size * num_sentences,))
        tokens0 = torch.nn.utils.rnn.pack_padded_sequence(tokens0, lengths=ws_lens0, batch_first=True, enforce_sorted=False)
        tokens1 = torch.nn.utils.rnn.pack_padded_sequence(tokens1, lengths=ws_lens1, batch_first=True, enforce_sorted=False)

        # pass the combined embedding through the word to sentence lstm
        tokens0, _ = self.word_to_sentence_lstm(tokens0)
        tokens1, _ = self.word_to_sentence_lstm(tokens1)

        # unpack the sequence
        tokens0, _ = torch.nn.utils.rnn.pad_packed_sequence(tokens0, batch_first=True, padding_value=0,
                                                            total_length=num_words)
        tokens1, _ = torch.nn.utils.rnn.pad_packed_sequence(tokens1, batch_first=True, padding_value=0,
                                                            total_length=num_words)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # try max over time instead of attention for testing
        # print(tokens0.shape)
        # tokens0 = torch.permute(tokens0, (0, 2, 1))
        # tokens1 = torch.permute(tokens1, (0, 2, 1))
        # # # # print(tokens0.shape)
        # # #
        # tokens0 = self.max_over_time2(tokens0)
        # tokens1 = self.max_over_time2(tokens1)
        # # # # print(tokens0.shape)
        # # #
        # tokens0 = torch.squeeze(tokens0)
        # tokens1 = torch.squeeze(tokens1)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        tokens0 = self.w2s_att(tokens0, ws_do_mask0)
        tokens1 = self.w2s_att(tokens1, ws_do_mask1)

        tokens0 = torch.reshape(tokens0, shape=(batch_size, num_sentences, -1))
        tokens1 = torch.reshape(tokens1, shape=(batch_size, num_sentences, -1))

        tokens0 = self.sentence_to_doc_lstm_do(tokens0)
        tokens1 = self.sentence_to_doc_lstm_do(tokens1)

        tokens0 = torch.nn.utils.rnn.pack_padded_sequence(tokens0, lengths=sd_lens0, batch_first=True, enforce_sorted=False)
        tokens1 = torch.nn.utils.rnn.pack_padded_sequence(tokens1, lengths=sd_lens1, batch_first=True, enforce_sorted=False)

        tokens0, _ = self.sentence_to_doc_lstm(tokens0)
        tokens1, _ = self.sentence_to_doc_lstm(tokens1)

        tokens0, _ = torch.nn.utils.rnn.pad_packed_sequence(tokens0, batch_first=True, padding_value=0,
                                                            total_length=num_sentences)
        tokens1, _ = torch.nn.utils.rnn.pad_packed_sequence(tokens1, batch_first=True, padding_value=0,
                                                            total_length=num_sentences)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # tokens0 = torch.permute(tokens0, (0, 2, 1))
        # tokens1 = torch.permute(tokens1, (0, 2, 1))
        # # print(tokens0.shape)
        #
        # tokens0 = self.max_over_time3(tokens0)
        # tokens1 = self.max_over_time3(tokens1)
        # # print(tokens0.shape)
        #
        # tokens0 = torch.squeeze(tokens0)
        # tokens1 = torch.squeeze(tokens1)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        tokens0 = self.s2d_att(tokens0, sd_do_mask0)
        tokens1 = self.s2d_att(tokens1, sd_do_mask1)

        # now the metric layer
        tokens0 = self.metric_do(tokens0)
        tokens1 = self.metric_do(tokens1)

        tokens0 = self.metric(tokens0)
        tokens1 = self.metric(tokens1)

        return tokens0, tokens1


def train_epoch(model,
                train_loader,
                optimizer,
                loss_fn,
                lr_scheduler=None,
                auth_prof_loader=None,
                eval_loader=None,
                eval_steps=-1,
                logging_steps=2000,
                max_grad_norm=1,
                epoch_idx=0,
                use_cuda=False,
                device=-1,
                use_wandb=False,
                logging_prefix='',
                save_best_model=False,
                save_model_checkpoints=False,
                save_model_steps=2000,
                save_model_dir='adhom_test_models'):

    model.train()

    running_loss = 0
    running_predictions = []
    running_labels = []
    step = None
    best_model_auc = 0
    loader_len = len(train_loader)

    pbar = tqdm(train_loader)

    for i, data in enumerate(pbar):

        step = epoch_idx * loader_len + i + 1

        labels, tokens0, tokens1, chars0, chars1, ws_do_mask0, ws_do_mask1, sd_do_mask0, sd_do_mask1, ws_lens0, ws_lens1, sd_lens0, sd_lens1 = data_to_device(data, use_cuda, device)
        optimizer.zero_grad()

        embedding0, embedding1 = model(tokens0, tokens1, chars0, chars1, ws_do_mask0, ws_do_mask1, sd_do_mask0, sd_do_mask1, ws_lens0, ws_lens1, sd_lens0, sd_lens1)

        # calculate loss
        loss = loss_fn(embedding0, embedding1, labels)
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # take update step
        optimizer.step()

        # track data
        running_loss += loss.item()

        # predictions
        running_predictions.extend(kernel_fn(euclidean_dist(embedding0, embedding1)).tolist())
        running_labels.extend(labels.tolist())

        if step % logging_steps == 0:
            last_loss = running_loss / logging_steps  # loss per batch
            # average accuracy per batch
            running_labels = [1 if x == 1 else 0 for x in running_labels]
            try:
                accuracy = av_metrics(running_labels, probas=running_predictions, threshold=0.5)['accuracy']
            except ValueError as e:
                accuracy = -1
            running_predictions = []
            running_labels = []
            pbar.set_description(f'avg loss: {last_loss:.4f}, acc: {accuracy:.4f}')
            if use_wandb:
                wandb.log({'Loss/train': last_loss}, step=step)
                wandb.log({'accuracy/train': accuracy}, step=step)
            running_loss = 0.

        if save_model_checkpoints:
            if step % save_model_steps == 0:
                save_model_to_disk(model, save_model_dir, best_model=False, step=step)

        if eval_steps > -1 and step % eval_steps == 0 and eval_loader is not None:
            if auth_prof_loader is None:
                logging.info(f'step {step}, launching evaluation')
                results = evaluate_model(model=model,
                                         val_loader=eval_loader,
                                         loss_fn=loss_fn,
                                         use_cuda=use_cuda,
                                         device=device,
                                         use_wandb=use_wandb,
                                         step=step)
                if save_best_model and results['auc'] > best_model_auc:
                    save_model_to_disk(model, optimizer, lr_scheduler, save_model_dir, best_model=True, step=step)
                    best_model_auc = results['auc']
            else:
                logging.info(f'step {step}, launching AA evaluation')
                results = evaluate_model_AA(model=model,
                                            auth_prof_loader=auth_prof_loader,
                                            val_loader=eval_loader,
                                            use_cuda=use_cuda,
                                            device=device,
                                            use_wandb=use_wandb,
                                            step=step)
                if save_best_model and results['macro_accuracy'] > best_model_auc:
                    save_model_to_disk(model, optimizer, lr_scheduler, save_model_dir, best_model=True, step=step)
                    best_model_auc = results['macro_accuracy']

    if lr_scheduler is not None:
        lr_scheduler.step()

    # return the step idx
    return model, optimizer, lr_scheduler, step, running_loss


def data_to_device(data, use_cuda, device):
    labels, tokens0, tokens1, chars0, chars1, ws_do_mask0, ws_do_mask1, sd_do_mask0, sd_do_mask1, ws_lens0, ws_lens1, sd_lens0, sd_lens1 = data
    if use_cuda:
        labels, tokens0, tokens1, = labels.to(device), tokens0.to(device), tokens1.to(device)
        chars0, chars1 = chars0.to(device), chars1.to(device)
        ws_do_mask0, ws_do_mask1 = ws_do_mask0.to(device), ws_do_mask1.to(device)
        sd_do_mask0, sd_do_mask1 = sd_do_mask0.to(device), sd_do_mask1.to(device)
    return labels, tokens0, tokens1, chars0, chars1, ws_do_mask0, ws_do_mask1, sd_do_mask0, sd_do_mask1, ws_lens0, ws_lens1, sd_lens0, sd_lens1


def evaluate_model(model,
                   val_loader,
                   loss_fn,
                   use_cuda=False,
                   device=-1,
                   use_wandb=False,
                   step=0):
    model.eval()

    with torch.no_grad():
        total_loss = 0

        all_labels = []
        all_probas = []

        eval_steps = 0

        for i, data in enumerate(tqdm(val_loader, desc='evaluating')):
            eval_steps += 1

            labels, tokens0, tokens1, chars0, chars1, ws_do_mask0, ws_do_mask1, sd_do_mask0, sd_do_mask1, ws_lens0, ws_lens1, sd_lens0, sd_lens1 = data_to_device(
                data, use_cuda, device)

            embedding0, embedding1 = model(tokens0, tokens1, chars0, chars1, ws_do_mask0, ws_do_mask1, sd_do_mask0,
                                           sd_do_mask1, ws_lens0, ws_lens1, sd_lens0, sd_lens1)
            sims = kernel_fn(euclidean_dist(embedding0, embedding1))

            # calculate loss
            total_loss += loss_fn(embedding0, embedding1, labels).item()

            # dists = cos_dist(embedding0, embedding1)
            # sims = dists
            all_probas.extend(sims.tolist())
            all_labels.extend(labels.tolist())

        _, results = grid_search(all_labels, all_probas)
        results.update({'avg_loss': total_loss / eval_steps})

        if use_wandb:
            wandb.log(results, step=step)

    model.train()
    return results


def evaluate_model_AA(model,
                      auth_prof_loader,
                      val_loader,
                      use_wandb=False,
                      use_cuda=False,
                      device=-1,
                      step=0):
    # so first we need to build our author embeddings
    # just going to do this in the same way that it is evaluated in AV problems
    model.eval()
    with torch.no_grad():
        auth_embeddings = {}
        for data in tqdm(auth_prof_loader, desc='building author profiles'):

            auth_id, tokens0, _, chars0, _, ws_do_mask0, _, sd_do_mask0, _, ws_lens0, _, sd_lens0, _ = data_to_device(data, use_cuda, device)

            auth_id = int(auth_id.tolist()[0])

            if auth_id in auth_embeddings:
                if len(auth_embeddings[auth_id]) >= 1:
                    continue

            # only need one embedding so only pass those toks
            embedding, _ = model(tokens0, tokens0, chars0, chars0, ws_do_mask0, ws_do_mask0, sd_do_mask0,
                                 sd_do_mask0, ws_lens0, ws_lens0, sd_lens0, sd_lens0)

            # set this embedding as the auth emb
            # auth_embeddings.setdefault(auth_id, []).append(embedding)
            auth_embeddings[auth_id] = embedding

        # for auth_id, embs in auth_embeddings.items():
        #     auth_embeddings[auth_id] = torch.unsqueeze(torch.mean(torch.cat(embs, dim=0), dim=0), dim=0)

        # now for the test dataset, compare auth embeddings to authors
        total_loss = 0
        all_preds = []
        all_labels = []
        all_sims = []
        # i think this works but only for batch size of 1. . .
        for data in tqdm(val_loader, desc='evaluating AA'):
            label, tokens0, _, chars0, _, ws_do_mask0, _, sd_do_mask0, _, ws_lens0, _, sd_lens0, _ = data_to_device(data, use_cuda, device)

            # only need one embedding so only pass those toks
            test_point_embedding, _ = model(tokens0, tokens0, chars0, chars0, ws_do_mask0, ws_do_mask0, sd_do_mask0,
                                            sd_do_mask0, ws_lens0, ws_lens0, sd_lens0, sd_lens0)

            most_similar_author = [[-1], -1]
            for auth_counter, (compared_auth_id, auth_embedding) in enumerate(auth_embeddings.items()):
                sim = kernel_fn(euclidean_dist(test_point_embedding, auth_embedding)).item()
                pred, pred_auth = sim, label[0]
                # logging.info(f'auth number: {auth_counter}, similarity: {sim}')
                if pred > most_similar_author[1]:
                    most_similar_author = [compared_auth_id, pred]

            # dists = cos_dist(embedding0, embedding1)
            # sims = dists
            all_preds.append(most_similar_author[0])
            all_labels.append(label.tolist()[0])
            all_sims.append(most_similar_author[1])

        # now calculate results
        results = aa_metrics(labels=all_labels, predictions=all_preds, raw_outputs=None, no_auc=True, prefix='')

        sim_stats = {
            'sim/avg': np.mean(all_sims),
            'sim/max': np.max(all_sims),
            'sim/min': np.min(all_sims),
            'sim/std': np.std(all_sims),
            'sim/median': np.median(all_sims)
        }

        results.update(sim_stats)

        if use_wandb:
            wandb.log(results, step=step)
    model.train()

    return results


def save_model_to_disk(model, optimizer, scheduler, save_dir, best_model, step):
    if best_model:
        path = os.path.join(save_dir, 'best_model', 'model.torch')
        logging.info(f'saving best model to {path}')
    else:
        path = os.path.join(save_dir, f'checkpoint-{step}', 'model.torch')
        logging.info(f'saving model checkpoint to {path}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict()
    }
    torch.save(save_dict, path)


def load_model_from_disk(path, model, optimizer=None, lr_scheduler=None):
    logging.info(f'loading model from {path}')
    torch_loaded = torch.load(path)
    model.load_state_dict(torch_loaded['model'])
    if optimizer is not None:
        optimizer.load_state_dict(torch_loaded['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(torch_loaded['lr_scheduler'])
    return model


def grid_search(labels, probs):
    # this isn't actually a grid search, don't need for now
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    # normalize labels
    labels[labels == -1] = 0

    sim_stats = {
        f'similarity_min': np.min(probs),
        f'similarity_max': np.max(probs),
        f'similarity_mean': np.mean(probs),
        f'similarity_std': np.std(probs),
        f'similarity_median': np.median(probs)
    }
    results = av_metrics(labels, probas=probs, threshold=0.5)
    results.update(sim_stats)

    return [0.5, results]


def kernel_fn(dist):
    return 1e-10 + torch.exp(-0.09 * torch.pow(dist, 3))


def euclidean_dist(emb0, emb1):
    return torch.diagonal(torch.cdist(emb0, emb1, p=2))


def cos_dist(emb0, emb1):
    return (torch.nn.CosineSimilarity()(emb0, emb1) + 1) / 2


def modified_contrastive_loss(emb0, emb1, labels, same_margin=0.95, diff_margin=0.05, dist_fn_name='euclidean'):
    # this works on 0, 1 labels, not -1, 1
    lbls = torch.clone(labels)
    lbls[lbls == -1] = 0

    if dist_fn_name == 'euclidean':
        dists = euclidean_dist(emb0, emb1)
        similarities = kernel_fn(dists)
    elif dist_fn_name == 'cosine':
        similarities = (torch.nn.CosineSimilarity()(emb0, emb1) + 1) / 2
    else:
        assert False, f'the distance function is not defined: {dist_fn_name}'

    loss = torch.mean(lbls * torch.pow(torch.relu(same_margin - similarities), 2) +
                      (1 - lbls) * torch.pow(torch.relu(similarities - diff_margin), 2))

    return loss


def main(args):
    # project management
    wandb_project = args.wandb_project
    use_wandb = False if wandb_project is None else True

    if use_wandb:
        args.save_model_dir = os.path.join(args.save_model_dir, wandb.run.name)
    else:
        args.save_model_dir = os.path.join(args.save_model_dir, 'test')

    # device
    device = args.device
    use_cuda = True if device >= 0 else False

    # logging
    logging_steps = args.logging_steps
    logging_prefix = 'test' if 'test' in args.test_path else 'val'
    # noinspection DuplicatedCode
    save_best_model = args.save_best_model
    save_model_checkpoint = args.save_model_checkpoints
    save_model_dir = args.save_model_dir

    # data
    train_path = args.train_path
    test_path = args.test_path
    num_dataloader_workers = args.num_dataloader_workers

    # training
    epochs = args.epochs
    model_path = args.model_path
    train_batch_size = args.train_batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    # noinspection DuplicatedCode
    loss_margin = args.loss_margin
    chr_vocab_size = args.chr_vocab_size
    tok_vocab_size = args.tok_vocab_size
    dont_use_fasttext = args.dont_use_fasttext
    max_chars_per_word = args.max_chars_per_word
    max_words_per_sentence = args.max_words_per_sentence
    max_sentences_per_doc = args.max_sentences_per_doc
    max_grad_norm = args.max_grad_norm
    lr_decay_gamma = args.lr_decay_gamma
    chr_count_min = args.chr_count_min
    tok_count_min = args.tok_count_min
    tok_file = args.tok_file
    chr_file = args.chr_file

    # evaluation
    evaluate_every_epoch = args.evaluate_every_epoch
    test_batch_size = args.test_batch_size
    evaluation_steps = args.evaluation_steps
    AA = args.AA

    if (save_model_checkpoint and evaluate_every_epoch) and (evaluation_steps < 0):
        logging.warning('you have selected to save model checkpoints, but not provided an evaluation step counter '
                        'model checkpoints will not be saved')

    if save_best_model and evaluate_every_epoch and evaluation_steps > -1:
        logging.warning('you have selected to evaluate every {evaluation_steps} steps and at the end of every epoch,'
                        'this will cause the best model to not be saved properly, please pick one evaluation period')

    # determine the exact char and token vocabularies we want to use based on counts and size
    # get the char and token counts
    tok_file = tok_file if tok_file is not None else f'{train_path}.adhom.tok_count'
    logging.info(f'getting token count file from {tok_file}')
    with open(tok_file, 'r') as tok_counts_file:
        tok_counts = json.load(tok_counts_file)

    chr_file = chr_file if chr_file is not None else f'{train_path}.adhom.chr_count'
    logging.info(f'getting character count file from {chr_file}')
    with open(chr_file, 'r') as char_counts_file:
        char_counts = json.load(char_counts_file)

    logging.info('building vocabularies from the count files')
    char_vocab = [[char, int(char_count)] for char, char_count in char_counts.items() if int(char_count) >= chr_count_min]
    char_vocab = OrderedDict([(x[0], x[1]) for x in sorted(char_vocab, key=lambda x: x[1], reverse=True)[:chr_vocab_size]])

    tok_vocab = [[tok, int(tok_count)] for tok, tok_count in tok_counts.items() if int(tok_count) >= tok_count_min]
    tok_vocab = OrderedDict([(x[0], x[1]) for x in sorted(tok_vocab, key=lambda x: x[1], reverse=True)[:tok_vocab_size]])

    if len(char_vocab) != args.chr_vocab_size:
        logging.warning(f'char vocab is being shorted to {len(char_vocab)} due to removing chars with < {chr_count_min} occurances')
        args.chr_vocab_size = len(char_vocab)

    if len(tok_vocab) != args.tok_vocab_size:
        logging.warning(f'token vocab is being shorted to {len(tok_vocab)} due to removing tokens with < {tok_count_min} occurences')
        args.tok_vocab_size = len(tok_vocab)

    # get the datasets, build the char and token maps from training data, and use the same maps for the test set
    logging.info('building the training and testing datasets')

    auth_profile_dataloader = None

    train_dataset = AADataset(train_path, char_vocab=char_vocab, tok_vocab=tok_vocab,
                              max_chars_per_word=max_chars_per_word, max_words_per_sentence=max_words_per_sentence,
                              max_sentences_per_doc=max_sentences_per_doc,
                              dont_use_fasttext=dont_use_fasttext)
    # logging.warning(f'!!!!!!!!!!!!! using AV train set for testing!!!!!!!!!!')
    # train_dataset = AVDataset(train_path, char_vocab=char_vocab, tok_vocab=tok_vocab,
    #                           max_chars_per_word=max_chars_per_word, max_words_per_sentence=max_words_per_sentence,
    #                           max_sentences_per_doc=max_sentences_per_doc,
    #                           dont_use_fasttext=dont_use_fasttext)

    if AA:
        logging.info('expecting to evaluate on an AA dataset')
        assert test_batch_size == 1, 'test_batch_size must be 1 to evaluate on an AA dataset'
        # build a dataloader for creating the author profiles
        auth_profile_dataset = AA_eval_dataset(train_path, char_vocab=char_vocab, tok_vocab=tok_vocab,
                                               max_chars_per_word=max_chars_per_word,
                                               max_words_per_sentence=max_words_per_sentence,
                                               max_sentences_per_doc=max_sentences_per_doc,
                                               dont_use_fasttext=dont_use_fasttext)
        auth_profile_dataloader = DataLoader(auth_profile_dataset, batch_size=1, shuffle=False,
                                             num_workers=num_dataloader_workers)

        test_dataset = AA_eval_dataset(test_path, char_vocab=char_vocab, tok_vocab=tok_vocab,
                                       max_chars_per_word=max_chars_per_word,
                                       max_words_per_sentence=max_words_per_sentence,
                                       max_sentences_per_doc=max_sentences_per_doc,
                                       dont_use_fasttext=dont_use_fasttext)

    else:
        test_dataset = AVDataset(test_path, char_vocab=char_vocab, tok_vocab=tok_vocab,
                                 max_chars_per_word=max_chars_per_word, max_words_per_sentence=max_words_per_sentence,
                                 max_sentences_per_doc=max_sentences_per_doc,
                                 dont_use_fasttext=dont_use_fasttext)
    # ensure the char transform is the same for the training and testing sets - tok is fine because of fasttext
    test_dataset.set_char_transform(train_dataset.get_char_transform())

    logging.info('building the dataloaders')
    # build dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                  num_workers=num_dataloader_workers)  #, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                                 num_workers=num_dataloader_workers)

    logging.info('building the model')
    # build a adhominem model
    model = AdHominem(args)
    if args.model_path is not None:
        model = load_model_from_disk(model_path, model)
    if use_cuda:
        logging.info(f'using cuda: device {device}')
        model.to(device=device)

    # get the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # get the lr schedular, using the fn from adhom for now
    # def lr_scheduler_fn(x, l=lr, epchs=epochs):
    #     if x < 2:
    #         return lr
    #     new_lr = min(l / ((1 + 5 * (x / epchs)) ** 0.4), 0.0015)
    #     old_lr = min(l / ((1 + 5 * ((x-1) / epchs)) ** 0.4), 0.0015)
    #     return new_lr / old_lr
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler_fn)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay_gamma)

    # get the loss function
    # this isn't quite the loss they use but we are going to experiment with it
    # loss_fn = nn.CosineEmbeddingLoss(margin=loss_margin)
    # this is the actual loss function used
    # TODO: !!!!! If i use euclidean distance here, it does not work, cos seems to not be super effective either
    # loss_fn = functools.partial(modified_contrastive_loss, same_margin=0.95, diff_margin=0.05, dist_fn_name='euclidean')
    loss_fn = modified_contrastive_loss

    logging.info('starting training:')
    best_model_auc = 0
    epochs_since_improvement = 0
    # train the model
    for epoch in tqdm(range(epochs), desc='epoch'):

        # if epoch > 0 and epoch % 5 == 0:
        #     try:
        #         train_dataloader.dataset.reset_sampled_data()
        #         logging.info('resetting training set sampled pairs')
        #     except Exception as e:
        #         logging.warning(f'problem trying to reset the sampled data in the training dataset: {e}')

        model, optimizer, lr_scheduler, step, _ = train_epoch(model=model,
                                                              train_loader=train_dataloader,
                                                              optimizer=optimizer,
                                                              lr_scheduler=lr_scheduler,
                                                              loss_fn=loss_fn,
                                                              auth_prof_loader=auth_profile_dataloader,
                                                              eval_loader=test_dataloader,
                                                              eval_steps=evaluation_steps,
                                                              logging_steps=logging_steps,
                                                              max_grad_norm=max_grad_norm,
                                                              epoch_idx=epoch,
                                                              use_cuda=use_cuda,
                                                              device=device,
                                                              use_wandb=use_wandb,
                                                              logging_prefix=logging_prefix,
                                                              save_model_dir=save_model_dir,
                                                              save_best_model=save_best_model)

        if evaluate_every_epoch:
            if args.AA:
                eval_results = evaluate_model_AA(model=model,
                                                 auth_prof_loader=auth_profile_dataloader,
                                                 val_loader=test_dataloader,
                                                 use_wandb=use_wandb,
                                                 use_cuda=use_cuda,
                                                 device=device,
                                                 step=step)
                if save_best_model and eval_results['macro_accuracy'] > best_model_auc:
                    save_model_to_disk(model, optimizer, lr_scheduler, save_model_dir, best_model=True, step=step)
                if eval_results['macro_accuracy'] <= best_model_auc:
                    logging.info('no performance improvement, watching for early stopping.')
                    epochs_since_improvement += 1
                else:
                    logging.info('improved accuracy, resetting early stopping criteria')
                    epochs_since_improvement = 0
            else:
                eval_results = evaluate_model(model=model,
                                              val_loader=test_dataloader,
                                              loss_fn=loss_fn,
                                              use_cuda=use_cuda,
                                              device=device,
                                              use_wandb=use_wandb,
                                              step=step)

                if save_best_model and eval_results['auc'] > best_model_auc:
                    save_model_to_disk(model, optimizer, lr_scheduler, save_model_dir, best_model=True, step=step)
                if eval_results['auc'] <= best_model_auc:
                    logging.info('no performance improvement, watching for early stopping.')
                    epochs_since_improvement += 1
                else:
                    logging.info('improved accuracy, resetting early stopping criteria')
                    epochs_since_improvement = 0

            if epochs_since_improvement > 4:
                logging.info('no improvement for 4 epochs, exiting.')
                break

    logging.info('Done!')


def torched_adhom_sweep(config=None, wandb_project=None, device=0, train_path=None, test_path=None):

    with wandb.init(config=config):
        config = wandb.config
        config.device = device
        config.train_path = train_path
        config.test_path = test_path
        config.wandb_project = wandb_project
        config.model = 'torched_adhom'
        wandb.config.update(config)
        main(config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # project management
    parser.add_argument('--wandb_project', type=str, default=None)

    # device
    parser.add_argument('--device', type=int, default=-1)

    # logging and saving
    parser.add_argument('--logging_steps', type=int, default=2000)
    parser.add_argument('--save_best_model', action='store_true')
    parser.add_argument('--save_model_checkpoints', action='store_true')
    parser.add_argument('--save_model_dir', type=str, default='adhominem_models')

    # data
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--num_dataloader_workers', type=int, default=10)

    # training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--loss_margin', type=float, default=0.05)
    parser.add_argument('--cnn_stride', type=int, default=1)
    parser.add_argument('--D_c', type=int, default=10)
    parser.add_argument('--D_r', type=int, default=30)
    parser.add_argument('--w', type=int, default=4)
    parser.add_argument('--D_w', type=int, default=300)
    parser.add_argument('--D_s', type=int, default=50)
    parser.add_argument('--D_d', type=int, default=50)
    parser.add_argument('--D_mlp', type=int, default=60)
    parser.add_argument('--max_chars_per_word', type=int, default=15)
    parser.add_argument('--max_words_per_sentence', type=int, default=30)
    parser.add_argument('--max_sentences_per_doc', type=int, default=50)
    parser.add_argument('--cnn_dropout_prob', type=float, default=0.2)
    parser.add_argument('--w2s_dropout_prob', type=float, default=0.1)
    parser.add_argument('--w2s_att_dropout_prob', type=float, default=0.1)
    parser.add_argument('--s2d_dropout_prob', type=float, default=0.1)
    parser.add_argument('--s2d_att_dropout_prob', type=float, default=0.1)
    parser.add_argument('--metric_dropout_prob', type=float, default=0.2)
    parser.add_argument('--chr_vocab_size', type=int, default=250)
    parser.add_argument('--tok_vocab_size', type=int, default=5000)
    parser.add_argument('--dont_use_fasttext', action='store_true')
    parser.add_argument('--max_grad_norm', type=float, default=1)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.96)
    parser.add_argument('--chr_count_min', type=int, default=100)
    parser.add_argument('--tok_count_min', type=int, default=10)
    parser.add_argument('--tok_file', type=str, default=None)
    parser.add_argument('--chr_file', type=str, default=None)

    # evaluation
    parser.add_argument('--evaluate_every_epoch', action='store_true')
    parser.add_argument('--evaluation_steps', type=int, default=2000)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--AA', action='store_true')

    parser_args = parser.parse_args()

    if parser_args.wandb_project is not None:
        parser_args.model = 'torched_adhom'
        with wandb.init(project=parser_args.wandb_project, config=vars(parser_args)):
            main(parser_args)
    else:
        main(parser_args)
