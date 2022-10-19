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
from valla.methods.torched_AdHominem import build_map_from_vocab
import numpy as np
import random
import fasttext
import os

logging.basicConfig(level=logging.DEBUG)
csv.field_size_limit(sys.maxsize)
tokenizer = spacy.load('en_core_web_lg')

word_embeddings = fasttext.load_model(os.path.join('data', 'cc.en.300.bin'))


def get_char_and_tok_ids_from_doc(doc, PADDING_TOKEN, UNK_TOKEN, char_vocab, tok_vocab, char_to_id, use_fasttext,
                                  word_embeddings, tok_to_id, max_chars_per_word, max_words_per_sentence,
                                  max_sentences_per_doc):
    # txt is a list of sentences, so here is what we need
    chars = np.ones(dtype=np.int32, shape=(max_sentences_per_doc, max_words_per_sentence,
                                           max_chars_per_word))
    # set everything as the padding token so it implicitly deals with too short texts/words/whatever
    chars = chars * char_to_id[PADDING_TOKEN]

    if use_fasttext:
        tokens = np.zeros(dtype=np.float32, shape=(max_sentences_per_doc, max_words_per_sentence, 300))
    else:
        tokens = np.ones(dtype=np.int32, shape=(max_sentences_per_doc, max_words_per_sentence))
        tokens = tokens * tok_to_id[PADDING_TOKEN]

    for sentence_num, sentence in enumerate(doc):
        if sentence_num >= max_sentences_per_doc:
            break
        # for each token in the sentence
        for tok_num, tok in enumerate(sentence.split(' ')):
            if tok_num >= max_words_per_sentence:
                break
            if use_fasttext:
                if tok in tok_vocab:
                    this_tok_emb = word_embeddings[tok]
                else:
                    this_tok_emb = word_embeddings['-kathyVodka']
                tokens[sentence_num][tok_num] = this_tok_emb
            else:
                if tok in tok_vocab:
                    this_tok = tok_to_id[tok]
                else:
                    this_tok = tok_to_id[UNK_TOKEN]
                tokens[sentence_num][tok_num] = this_tok

            # for every character in the token
            for char_num, char in enumerate(tok):
                if char_num >= max_chars_per_word:
                    break
                if char in char_vocab:
                    this_char = char_to_id[char]
                else:
                    this_char = char_to_id[UNK_TOKEN]
                chars[sentence_num][tok_num][char_num] = this_char

    # tokens is of dimension (num_sentences, sentence_len)
    # chars is of dimension (num_sentences, num_tokens, token_len)
    return tokens, chars


def getitemAA(item, data, author_list, idx_to_txt_map, PADDING_TOKEN, UNK_TOKEN, char_vocab, tok_vocab, char_to_id,
              use_fasttext,
              tok_to_id, max_chars_per_word, max_words_per_sentence,
              max_sentences_per_doc):
    auth_id = idx_to_txt_map[item]['auth_id']
    txt_num = idx_to_txt_map[item]['text_id']
    # text0 is a list of sentences
    text0 = data[auth_id][txt_num]
    # return a randomly sampled tuple
    # if random.random() < 0.5:
    #     # different author sample
    #     label = -1
    #     auth2 = random.choice(author_list)
    #     while auth2 == auth_id:
    #         auth2 = random.choice(author_list)
    #     text1 = random.choice(data[auth2])
    #
    # else:
    #     label = 1
    #     text1 = random.choice(data[auth_id])
    #     # ignoring the chance that text0 and 1 are the same - loss will be zero so will essentially just be ignored

    tokens0, chars0 = get_char_and_tok_ids_from_doc(text0, PADDING_TOKEN, UNK_TOKEN, char_vocab, tok_vocab, char_to_id,
                                                    use_fasttext,
                                                    word_embeddings, tok_to_id, max_chars_per_word,
                                                    max_words_per_sentence,
                                                    max_sentences_per_doc)
    # tokens1, chars1 = get_char_and_tok_ids_from_doc(text1, PADDING_TOKEN, UNK_TOKEN, char_vocab, tok_vocab, char_to_id,
    #                                                 use_fasttext,
    #                                                 word_embeddings, tok_to_id, max_chars_per_word,
    #                                                 max_words_per_sentence,
    #                                                 max_sentences_per_doc)
    # so label is an int, text0 and text1 are a list of strings
    #  chars0 and chars1 is a list of lists of chars (one list of chars per sentense)
    return auth_id, tokens0, chars0  # label, tokens0, tokens1, chars0, chars1


def getitemAV(item, data, author_list, idx_to_txt_map, PADDING_TOKEN, UNK_TOKEN, char_vocab, tok_vocab, char_to_id,
              use_fasttext,
              tok_to_id, max_chars_per_word, max_words_per_sentence,
              max_sentences_per_doc):

    label, text0, text1 = data[item]
    label = label if label == 1 else -1
    tokens0, chars0 = get_char_and_tok_ids_from_doc(text0, PADDING_TOKEN, UNK_TOKEN, char_vocab, tok_vocab, char_to_id,
                                                    use_fasttext,
                                                    word_embeddings, tok_to_id, max_chars_per_word,
                                                    max_words_per_sentence,
                                                    max_sentences_per_doc)
    tokens1, chars1 = get_char_and_tok_ids_from_doc(text1, PADDING_TOKEN, UNK_TOKEN, char_vocab, tok_vocab, char_to_id,
                                                    use_fasttext,
                                                    word_embeddings, tok_to_id, max_chars_per_word,
                                                    max_words_per_sentence,
                                                    max_sentences_per_doc)
    return label, tokens0, tokens1, chars0, chars1


class PreTokenizeAdHominem:
    def __init__(self, data_path, save_path, is_av_dset, char_vocab, tok_vocab, char_to_id,
                 use_fasttext,
                 tok_to_id, max_chars_per_word, max_words_per_sentence,
                 max_sentences_per_doc, overwrite):
        self.data_path = data_path
        self.save_path = save_path
        self.is_av_dset = is_av_dset
        self.processed_samples = []
        self.tok_to_id = tok_to_id
        self.overwrite = overwrite

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if overwrite:
            if os.path.isfile(save_path):
                # delete the file
                os.remove(save_path)
        else:
            assert not os.path.isfile(save_path), 'the file at the given save location already exists, ' \
                                                  'must pass overwrite in to reprocess it'

        if is_av_dset:

            self.data = []

            with open(self.data_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self.data.append([int(row[0]), json.loads(row[1]), json.loads(row[2])])

            self.data_len = len(self.data)
            self.idx_to_txt_map = None
            self.author_list = None

        else:

            self.data = {}

            with open(self.data_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self.data.setdefault(int(row[0]), []).append(json.loads(row[1]))

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

            self.data_len = i
            self.author_list = list(self.data.keys())

        # always need an unknown token
        self.UNK_TOKEN = '<UNK>'
        self.PADDING_TOKEN = '<PAD>'
        self.max_chars_per_word = max_chars_per_word
        self.max_words_per_sentence = max_words_per_sentence
        self.max_sentences_per_doc = max_sentences_per_doc

        self.use_fasttext = use_fasttext

        # if self.use_fasttext:
        #     self.word_embeddings = fasttext.load_model(os.path.join('data', 'cc.en.300.bin'))

        if char_to_id is not None:
            assert self.UNK_TOKEN in char_to_id, 'the unknown token was not found in the given char-to-id map'
            self.char_to_id = char_to_id
            self.char_vocab = set(char_to_id.keys())
        else:
            assert char_vocab is not None, 'you did not provide a char-to-id map, so you must provide a char-vocabulary'
            self.char_vocab = set(char_vocab)
            self.char_to_id = build_map_from_vocab(self.char_vocab, self.UNK_TOKEN, self.PADDING_TOKEN)

        if not self.use_fasttext:
            if tok_to_id is not None:
                assert self.UNK_TOKEN in tok_to_id, 'the unknown token was not found in the given tok-to-id map'
                self.tok_to_id = tok_to_id
                self.tok_vocab = set(tok_to_id.keys())
            else:
                assert tok_vocab is not None, 'you did not provide a tok-to-id map, you must provide a tok-vocabulary'
                self.tok_vocab = set(tok_vocab)
                self.tok_to_id = build_map_from_vocab(self.tok_vocab, self.UNK_TOKEN, self.PADDING_TOKEN)
        else:
            self.tok_vocab = word_embeddings.get_words()

    def get_char_and_tok_maps(self):
        if self.use_fasttext:
            return self.char_to_id, None
        return self.char_to_id, self.tok_to_id,

    def update_self(self, label, tokens0, tokens1, chars0, chars1):
        # instead of saving, just make a jsonl file
        with open(self.save_path, 'a') as f:
            f.write(json.dumps([label, tokens0.tolist(), tokens1.tolist(), chars0.tolist(), chars1.tolist()]) + '\n')

    def update_self_aa(self, label, tokens, chars):
        with open(self.save_path, 'a') as f:
            f.write(json.dumps([label, tokens.tolist(), chars.tolist()]) + '\n')

    def tokenize_dataset(self, num_workers=10):
        async_results = {}

        logging.info(f'launching the preprocessing to tokenize the train dataset, using {num_workers} workers')
        with Pool(processes=num_workers) as pool:
            fn = getitemAV if self.is_av_dset else getitemAA

            fn = functools.partial(fn,
                                   data=self.data,
                                   author_list=self.author_list,
                                   idx_to_txt_map=self.idx_to_txt_map,
                                   PADDING_TOKEN=self.PADDING_TOKEN,
                                   UNK_TOKEN=self.UNK_TOKEN,
                                   char_vocab=self.char_vocab,
                                   tok_vocab=self.tok_vocab,
                                   char_to_id=self.char_to_id,
                                   use_fasttext=self.use_fasttext,
                                   tok_to_id=self.tok_to_id,
                                   max_chars_per_word=self.max_chars_per_word,
                                   max_words_per_sentence=self.max_words_per_sentence,
                                   max_sentences_per_doc=self.max_sentences_per_doc)

            for idx in range(self.data_len):
                async_results[idx] = pool.apply_async(fn, (idx,))

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
                        if self.is_av_dset:
                            self.update_self(*res)
                        else:
                            self.update_self_aa(*res)
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
                    logging.info(
                        f'Tokenizing: approximately {(res_left / ((num_removed+1) / 30)) / 60} minutes remaining')
                    last_check = time.time()
                    num_removed = 0

            logging.info('finished!')
            # # now save the tokenized dataset
            # save_path = self.save_path
            # logging.info(f'saving the tokenized training dataset to: {save_path}')
            # with open(save_path, 'w') as f:
            #     json.dump(self.processed_samples, f)
            # logging.info('done!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--AV', action='store_true')
    parser.add_argument('--tok_counts_file', type=str, default=None)
    parser.add_argument('--char_counts_file', type=str)
    parser.add_argument('--chr_vocab_size', type=int, default=250)
    parser.add_argument('--tok_vocab_size', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--max_chars_per_word', type=int, default=15)
    parser.add_argument('--max_words_per_sentence', type=int, default=30)
    parser.add_argument('--max_sentences_per_doc', type=int, default=50)
    parser.add_argument('--overwrite', action='store_true')
    parser.set_defaults(AV=False, overwrite=False)
    args = parser.parse_args()

    data_path = args.data_path
    tok_counts_file = args.tok_counts_file
    char_counts_file = args.char_counts_file
    num_workers = args.num_workers
    chr_vocab_size = args.chr_vocab_size
    tok_vocab_size = args.tok_vocab_size
    max_chars_per_word = args.max_chars_per_word
    max_words_per_sentence = args.max_words_per_sentence
    max_sentences_per_doc = args.max_sentences_per_doc
    is_av_dataset = args.AV

    logging.info('getting token and character count files')
    with open(char_counts_file, 'r') as cc_file:
        char_counts = json.load(cc_file)

    with open(tok_counts_file, 'r') as tc_file:
        tok_counts = json.load(tc_file)

    logging.info('building vocabularies from the count files')

    char_vocab = [[char, int(char_count)] for char, char_count in char_counts.items()]
    char_vocab = [x[0] for x in sorted(char_vocab, key=lambda x: x[1], reverse=True)[:chr_vocab_size]]

    tok_vocab = [[tok, int(tok_count)] for tok, tok_count in tok_counts.items()]
    tok_vocab = [x[0] for x in sorted(tok_vocab, key=lambda x: x[1], reverse=True)[:tok_vocab_size]]

    # if is_av_dataset:
    #     adhom_dset = AVDataset(data_path, char_vocab, tok_vocab, char_to_id=None, tok_to_id=None,
    #                            max_chars_per_word=max_chars_per_word,
    #                            max_words_per_sentence=max_words_per_sentence,
    #                            max_sentences_per_doc=max_sentences_per_doc,
    #                            dont_use_fasttext=False)
    # else:
    #     adhom_dset = AADataset(data_path, char_vocab, tok_vocab, char_to_id=None, tok_to_id=None,
    #                            max_chars_per_word=max_chars_per_word,
    #                            max_words_per_sentence=max_words_per_sentence,
    #                            max_sentences_per_doc=max_sentences_per_doc,
    #                            dont_use_fasttext=False)

    # now for every index in the dataset, launch a process to properly tokenize i think

    save_path = data_path + f'.{chr_vocab_size}_{max_chars_per_word}_{max_words_per_sentence}_{max_sentences_per_doc}.json'
    adhom_tokenizer = PreTokenizeAdHominem(data_path, save_path, is_av_dataset,
                                           char_vocab=char_vocab, tok_vocab=tok_vocab,
                                           char_to_id=None, tok_to_id=None,
                                           max_chars_per_word=max_chars_per_word,
                                           max_words_per_sentence=max_words_per_sentence,
                                           max_sentences_per_doc=max_sentences_per_doc,
                                           use_fasttext=True,
                                           overwrite=args.overwrite)

    logging.info('starting the tokenizing and counting')
    adhom_tokenizer.tokenize_dataset(num_workers)
    logging.info('finished')

if __name__ == '__main__':
    main()
