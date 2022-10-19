# adapted from https://github.com/janithnw/pan2020_authorship_verification

import pickle
import numpy as np
from tqdm.auto import trange, tqdm
import json
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV
import os
from textcomplexity import vocabulary_richness
import re
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from itertools import chain
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import conll2000
import nltk
import nltk.data
import itertools
import spacy
import argparse
import csv
import wandb

import valla.utils.eval_metrics
from valla.utils.misspelt_words_feature import MisspellingsFeatureTransformer
from valla.dsets.loaders import get_av_dataset
from collections import defaultdict

DATA_DIR = 'data/large/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
DATA_PATH = DATA_DIR + 'pan20-authorship-verification-training-large.jsonl'
TEMP_DATA_PATH = 'temp_data/large_model_training_data/'
PREPROCESSED_DATA_PATH = 'temp_data/large_model_training_data/'

dirname = os.path.dirname(__file__)
regex_chunker = None
ml_chunker = None
tnlp_regex_chunker = None

treebank_tokenizer = nltk.tokenize.TreebankWordTokenizer()

nlp_stanza = None  # stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)
nlp_spacy = None  # spacy.load("en_core_web_sm", disable=['ner'])

tagger = nltk.data.load(os.path.join(dirname, "pos_tagger/treebank_brill_aubt.pickle"))
perceptron_tagger = PerceptronTagger()

ground_truth = {}

grammar = r"""
  NP: 
      {<DT|WDT|PP\$|PRP\$>?<\#|CD>*(<JJ|JJS|JJR><VBG|VBN>?)*(<NN|NNS|NNP|NNPS>(<''><POS>)?)+}
      {<DT|WDT|PP\$|PRP\$><JJ|JJS|JJR>*<CD>}
      {<DT|WDT|PP\$|PRP\$>?<CD>?(<JJ|JJS|JJR><VBG>?)}
      {<DT>?<PRP|PRP\$>}
      {<WP|WP\$>}
      {<DT|WDT>}
      {<JJR>}
      {<EX>}
      {<CD>+}
  VP: {<VBZ><VBG>}
      {(<MD|TO|RB.*|VB|VBD|VBN|VBP|VBZ>)+}

"""

tweetNLP_grammar = r"""

    NP: {<X>?<D>?<\$>?<A>?(<R>?<A>)*<NOM>}
    NP: {(<O>|<\$>)+}         # Pronouns and propper nouns

    PP: {<P><NP>+}                 # Basic Prepositional Phrase
    PP: {<R|A>+<P><NP>+} 

    # Nominal is a noun, followed optionally by a series of post-modifiers
    # Post modifiers could be:
    # - Prepositional phrase
    # - non-finite postmodifiers (<V><NP>|<V><PP>|<V><NP><PP>)
    # - postnominal relative clause  (who | that) VP 
    NOM: {<L|\^|N>+(<PP>|<V><NP>|<V><PP>|<V><NP><PP>|<P|O><VP>)+}
    NOM: {<L|\^|N>+}
    NP: {<NP><\&><NP>}

    VP: {<R>*<V>+(<NP>|<PP>|<NP><PP>)+}
    VP: {<VP><\&><VP>}
"""


def merge_entries(entries):
    ret = {}
    for k in ['pos_tags', 'pos_tag_chunks', 'pos_tag_chunk_subtrees', 'tokens']:
        l = [e[k] for e in entries]
        ret[k] = list(itertools.chain.from_iterable(l))
    ret['preprocessed'] = '\n'.join([e['preprocessed'] for e in entries])
    return ret


# https://medium.com/analytics-vidhya/basic-tweet-preprocessing-method-with-python-56b4e53854a1
def preprocess_tweet(text):
    # remove URLs
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' URL ', text)
    text = re.sub(r'http\S+', '', text)
    # remove usernames
    text = re.sub('@[^\s]+', '@USER', text)

    return text


def preprocess_text(text):
    # remove URLs
    text = text.lower()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' URL ', text)

    return text


def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
        histo = "<START>"
    else:
        prevword, prevpos = sentence[i - 1]
        histo = history[-1]
    if i == len(sentence) - 1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i + 1]
    return {"pos": pos,
            "word": word,
            "hist": histo,
            "prevpos": prevpos,
            "nextpos": nextpos,
            "prevpos+pos": "%s+%s" % (prevpos, pos),
            "pos+nextpos": "%s+%s" % (pos, nextpos)
            }


class ConsecutiveNPChunkTagger(nltk.TaggerI):  # [_consec-chunk-tagger]

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)  # [_consec-use-fe]
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(  # [_consec-use-maxent]
            train_set, algorithm='IIS', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):  # [_consec-chunker]
    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c) for (w, t, c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


def train_chunker():
    train_sents = conll2000.chunked_sents('train.txt')
    test_sents = conll2000.chunked_sents('test.txt')
    chunker = ConsecutiveNPChunker(train_sents)
    print(chunker.evaluate(test_sents))
    pickle.dump(chunker, open(os.path.join(dirname, 'temp_data/chunker.p'), 'wb'))


def get_nltk_pos_tag_based_regex_chunker():
    global regex_chunker
    if regex_chunker is not None:
        return regex_chunker
    regex_chunker = nltk.RegexpParser(grammar)
    return regex_chunker


def get_tweet_nlp_pos_tag_based_regex_chunker():
    global tnlp_regex_chunker
    if tnlp_regex_chunker is not None:
        return tnlp_regex_chunker
    tnlp_regex_chunker = nltk.RegexpParser(tweetNLP_grammar)
    return tnlp_regex_chunker


def get_nltk_pos_tag_based_ml_chunker():
    global ml_chunker
    if ml_chunker is not None:
        return ml_chunker
    if os.path.isfile(os.path.join(dirname, 'temp_data/chunker.p')):
        ml_chunker = pickle.load(open(os.path.join(dirname, 'temp_data/chunker.p'), 'rb'))
        return ml_chunker
    print('Training Chunker...')
    train_chunker()
    return ml_chunker


def chunk_to_str(chunk):
    if type(chunk) is nltk.tree.Tree:
        return chunk.label()
    else:
        return chunk[1]


def extract_subtree_expansions(t, res):
    if type(t) is nltk.tree.Tree:
        expansion = t.label() + "[" + " ".join([chunk_to_str(child) for child in t]) + "]"
        res.append(expansion)
        for child in t:
            extract_subtree_expansions(child, res)


def pos_tag_chunk(pos_tags, chunker):
    parse_tree = chunker.parse(pos_tags)
    subtree_expansions = []
    for subt in parse_tree:
        extract_subtree_expansions(subt, subtree_expansions)
    return list(map(chunk_to_str, parse_tree)), subtree_expansions


def tokenize(text, tokenizer):
    if tokenizer == 'treebank':
        return treebank_tokenizer.tokenize(text)
    if tokenizer == 'casual':
        return nltk.tokenize.casual_tokenize(text)
    if tokenizer == 'spacy':
        return map(lambda t: t.text, nlp_spacy(text))
    if tokenizer == 'stanza':
        return map(lambda t: t.text, nlp_stanza(text).iter_tokens())
    raise 'Unknown tokenizer type. Valid options: [treebank, casual, spacy, stanza]'


def prepare_entry(text, mode=None, tokenizer='treebank'):
    tokens = []
    # Workaround because there re some docuemtns that are repitions of the same word which causes the regex chunker to hang
    prev_token = ''
    # for t in tokenizer.tokenize(text):
    for t in tokenize(text, tokenizer):
        if t != prev_token:
            tokens.append(t)
    if mode is None or mode == 'fast':
        tagger_output = tagger.tag(tokens)
        pos_tags = [t[1] for t in tagger_output]
        pos_chunks, subtree_expansions = pos_tag_chunk(tagger_output, get_nltk_pos_tag_based_regex_chunker())
    elif mode == 'accurate':
        tagger_output = perceptron_tagger.tag(tokens)
        pos_tags = [t[1] for t in tagger_output]
        pos_chunks, subtree_expansions = pos_tag_chunk(tagger_output, get_nltk_pos_tag_based_ml_chunker())

    entry = {
        'preprocessed': text,
        'pos_tags': pos_tags,
        'pos_tag_chunks': pos_chunks,
        'pos_tag_chunk_subtrees': subtree_expansions,
        'tokens': [preprocess_text(t) for t in tokens]
    }
    return entry


def prepare_tweets_entry(tweets, pos_tags_list):
    flattened_tags = [(t[0], t[1]) for t in chain.from_iterable(pos_tags_list)]

    tokens = [t[0] for t in flattened_tags]
    pos_tags = [t[1] for t in flattened_tags]

    pos_chunks = []
    subtree_expansions = []
    for pt in pos_tags_list:
        p, s = pos_tag_chunk([(t[0], t[1]) for t in pt], get_nltk_pos_tag_based_ml_chunker())
        pos_chunks.extend(p)
        subtree_expansions.extend(s)

    entry = {
        'preprocessed': '\n'.join([preprocess_tweet(t) for t in tweets]),
        'pos_tags': pos_tags,
        'pos_tag_chunks': pos_chunks,
        'pos_tag_chunk_subtrees': subtree_expansions,
        'tokens': tokens
    }
    return entry


def word_count(entry):
    return len(entry['tokens'])


def avg_chars_per_word(entry):
    r = np.mean([len(t) for t in entry['tokens']])
    return r


def distr_chars_per_word(entry, max_chars=10):
    counts = [0] * max_chars
    if len(entry['tokens']) == 0:
        return counts
    for t in entry['tokens']:
        l = len(t)
        if l <= max_chars:
            counts[l - 1] += 1
    r = [c / len(entry['tokens']) for c in counts]
    #     fnames = ['distr_chars_per_word_' + str(i + 1)  for i in range(max_chars)]
    return r


def character_count(entry):
    r = len(re.sub('\s+', '', entry['preprocessed']))
    return r


# def spell_err_freq(entry):


# https://github.com/ashenoy95/writeprints-static/blob/master/whiteprints-static.py
def hapax_legomena(entry):
    freq = nltk.FreqDist(word for word in entry['tokens'])
    hapax = [key for key, val in freq.items() if val == 1]
    dis = [key for key, val in freq.items() if val == 2]
    if len(dis) == 0 or len(entry['tokens']) == 0:
        return 0
    # return (len(hapax) / len(dis)) / len(entry['tokens'])
    return (len(hapax) / len(dis))


VOCAB_RICHNESS_FNAMES = ['type_token_ratio', 'guiraud_r', 'herdan_c', 'dugast_k', 'maas_a2', 'dugast_u', 'tuldava_ln',
                         'brunet_w', 'cttr', 'summer_s', 'sichel_s', 'michea_m', 'honore_h', 'herdan_vm', 'entropy',
                         'yule_k', 'simpson_d']


def handle_exceptions(func, *args):
    try:
        return func(*args)
    except:
        # print('Error occured', func, *args)
        return 0.0


def compute_vocab_richness(entry):
    if len(entry['tokens']) == 0:
        return np.zeros(len(VOCAB_RICHNESS_FNAMES))
    window_size = 1000
    res = []
    for chunk in chunker(entry['tokens'], window_size):
        text_length, vocabulary_size, frequency_spectrum = vocabulary_richness.preprocess(chunk, fs=True)
        res.append([
            handle_exceptions(vocabulary_richness.type_token_ratio, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.guiraud_r, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.herdan_c, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.dugast_k, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.maas_a2, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.dugast_u, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.tuldava_ln, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.brunet_w, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.cttr, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.summer_s, text_length, vocabulary_size),

            handle_exceptions(vocabulary_richness.sichel_s, vocabulary_size, frequency_spectrum),
            handle_exceptions(vocabulary_richness.michea_m, vocabulary_size, frequency_spectrum),

            handle_exceptions(vocabulary_richness.honore_h, text_length, vocabulary_size, frequency_spectrum),
            handle_exceptions(vocabulary_richness.herdan_vm, text_length, vocabulary_size, frequency_spectrum),

            handle_exceptions(vocabulary_richness.entropy, text_length, frequency_spectrum),
            handle_exceptions(vocabulary_richness.yule_k, text_length, frequency_spectrum),
            handle_exceptions(vocabulary_richness.simpson_d, text_length, frequency_spectrum),
        ])
    return np.array(res).mean(axis=0)


def pass_fn(x):
    return x


class CustomTfIdfTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, key, analyzer, n=1, vocab=None):
        self.key = key
        if self.key == 'pos_tags' or self.key == 'tokens' or self.key == 'pos_tag_chunks' or self.key == 'pos_tag_chunk_subtrees':
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1, tokenizer=pass_fn, preprocessor=pass_fn,
                                              vocabulary=vocab, norm='l2', ngram_range=(1, n))
        else:
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1, vocabulary=vocab, norm='l2',
                                              ngram_range=(1, n))

    def fit(self, x, y=None):
        self.vectorizer.fit([entry[self.key] for entry in x], y)
        return self

    def transform(self, x):
        return self.vectorizer.transform([entry[self.key] for entry in x])

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


class CustomFreqTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, analyzer, n=1, vocab=None):
        self.vectorizer = TfidfVectorizer(tokenizer=pass_fn, preprocessor=pass_fn, vocabulary=vocab, norm=None,
                                          ngram_range=(1, n))

    def fit(self, x, y=None):
        self.vectorizer.fit([entry['tokens'] for entry in x], y)
        return self

    def transform(self, x):
        d = np.array([1 + len(entry['tokens']) for entry in x])[:, None]
        return self.vectorizer.transform([entry['tokens'] for entry in x]) / d

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


class CustomFuncTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_func, fnames=None):
        self.transformer_func = transformer_func
        self.fnames = fnames

    def fit(self, x, y=None):
        return self;

    def transform(self, x):
        xx = np.array([self.transformer_func(entry) for entry in x])
        if len(xx.shape) == 1:
            return xx[:, None]
        else:
            return xx

    def get_feature_names(self):
        if self.fnames is None:
            return ['']
        else:
            return self.fnames


def get_stopwords():
    with open(os.path.join(dirname, 'stopwords.txt'), 'r') as f:
        words = [l.strip() for l in f.readlines()]
        return words


class MaskedStopWordsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords, n):
        self.stopwords = set(stopwords)
        self.vectorizer = TfidfVectorizer(tokenizer=pass_fn, preprocessor=pass_fn, min_df=0.1, ngram_range=(1, n))

    def _process(self, entry):
        return [
            entry['tokens'][i] if entry['tokens'][i] in self.stopwords else entry['pos_tags'][i]
            for i in range(len(entry['tokens']))
        ]

    def fit(self, X, y=None):
        X = [self._process(entry) for entry in X]
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        X = [self._process(entry) for entry in X]
        return self.vectorizer.transform(X)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


class POSTagStats(BaseEstimator, TransformerMixin):
    POS_TAGS = [
        'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ',
        'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS',
        'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',
        'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH',
        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
        'WP', 'WP$', 'WRB'
    ]

    def __init__(self):
        pass

    def _process(self, entry):
        tags_dict = defaultdict(set)
        tags_word_length = defaultdict(list)
        for i in range(len(entry['tokens'])):
            tags_dict[entry['pos_tags'][i]].add(entry['tokens'][i])
            tags_word_length[entry['pos_tags'][i]].append(len(entry['tokens'][i]))
        res_tag_fractions = np.array([len(tags_dict[t]) for t in self.POS_TAGS])
        if res_tag_fractions.sum() > 0:
            res_tag_fractions = res_tag_fractions / res_tag_fractions.sum()

        res_tag_word_lengths = np.array(
            [np.mean(tags_word_length[t]) if len(tags_word_length[t]) > 0 else 0 for t in self.POS_TAGS])
        return np.concatenate([res_tag_fractions, res_tag_word_lengths])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._process(entry) for entry in X]

    def get_feature_names(self):
        return ['tag_fraction_' + t for t in self.POS_TAGS] + ['tag_word_length_' + t for t in self.POS_TAGS]


class DependencyFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, n=3):
        self.vectorizer = TfidfVectorizer(tokenizer=self.pass_fn, preprocessor=self.pass_fn)
        self.nlp = spacy.load("en_core_web_sm")

    def pass_fn(self, x):
        return x

    def extract_dep_n_grams(self, node, current_list, output, n, curr_depth=0):
        current_list.append(node.dep_)
        output.append(current_list)
        if curr_depth > 200:
            return
        for c in node.children:
            l = current_list.copy()
            if len(l) > n - 1:
                l = l[1:]
            self.extract_dep_n_grams(c, l, output, n, curr_depth + 1)

    def dep_ngrams_to_str(self, ngam_list):
        return ['_'.join(ngrams).lower() for ngrams in ngam_list]

    def process(self, text):
        dep_ngrams = []
        doc = self.nlp(text)
        for s in doc.sents:
            o = []
            self.extract_dep_n_grams(s.root, [], o, 4)
            dep_ngrams.extend(o)
        return self.dep_ngrams_to_str(dep_ngrams)

    def fit(self, x, y=None):
        xx = [self.process(entry['preprocessed']) for entry in x]
        self.vectorizer.fit(xx, y)
        return self

    def transform(self, x):
        xx = [self.process(entry['preprocessed']) for entry in x]
        return self.vectorizer.transform(xx)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


def get_transformer(selected_featuresets=None, char_n=3):
    char_distr = CustomTfIdfTransformer('preprocessed', 'char_wb', n=char_n)
    # word_distr = CustomTfIdfTransformer('preprocessed', 'word', n=3)
    # pos_tag_distr = CustomTfIdfTransformer('pos_tags', 'word', n=3)
    # pos_tag_chunks_distr = CustomTfIdfTransformer('pos_tag_chunks', 'word', n=3)
    # pos_tag_chunks_subtree_distr = CustomTfIdfTransformer('pos_tag_chunk_subtrees', 'word', n=1)
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{¦}~'
    # special_char_distr = CustomTfIdfTransformer('preprocessed', 'char_wb', vocab=punctuation)
    # freq_func_words = CustomFreqTransformer('word', vocab=get_stopwords())

    featuresets = [
        ('char_distr', char_distr),
        # ('word_distr', word_distr),
        # ('pos_tag_distr', pos_tag_distr),
        # ('pos_tag_chunks_distr', pos_tag_chunks_distr),
        # ('pos_tag_chunks_subtree_distr', pos_tag_chunks_subtree_distr),
        # ('special_char_distr', special_char_distr),
        # ('freq_func_words', freq_func_words),
        # ('hapax_legomena', CustomFuncTransformer(hapax_legomena)),
        # # ('character_count', CustomFuncTransformer(character_count)),
        # ('distr_chars_per_word', CustomFuncTransformer(distr_chars_per_word, fnames=[str(i) for i in range(10)])),
        # ('avg_chars_per_word', CustomFuncTransformer(avg_chars_per_word)),
        # # ('word_count', CustomFuncTransformer(word_count))
        # ('vocab_richness', CustomFuncTransformer(compute_vocab_richness, fnames=VOCAB_RICHNESS_FNAMES)),
        # ('misspellings', MisspellingsFeatureTransformer(data_dir=os.path.join(dirname, 'temp_data/misspellings/'))),
        # ('masked_stop_words_distr', MaskedStopWordsTransformer(get_stopwords(), 3)),
        # ('pos_tag_stats', POSTagStats())
    ]
    if selected_featuresets is None:
        transformer = FeatureUnion(featuresets)
    else:
        transformer = FeatureUnion([f for f in featuresets if f[0] in selected_featuresets])

    # pipeline = Pipeline([('features', transformer), ('selection', VarianceThreshold())])
    return transformer


def fit_transformers(data, char_n=3):  #, data_fraction=0.01):
    # docs_1 = []
    # docs_2 = []

    # with open(PREPROCESSED_DATA_PATH + 'preprocessed_train.jsonl', 'r') as f:
    #     for l in tqdm(f):
    #         if np.random.rand() < data_fraction:
    #             d = json.loads(l)
    #             docs_1.append(d['pair'][0])
    #             docs_2.append(d['pair'][1])

    transformer = get_transformer(char_n=char_n)
    scaler = StandardScaler()
    secondary_scaler = StandardScaler()

    X = transformer.fit_transform(data).todense()  # docs_1 + docs_2).todense()
    X = scaler.fit_transform(X)
    X1 = X[:int(len(X)/2)]
    X2 = X[int(len(X)/2):]
    secondary_scaler.fit(np.abs(X1 - X2))

    return transformer, scaler, secondary_scaler


def vectorize(XX, Y, ordered_idxs, transformer, scaler, secondary_scaler, preprocessed_path, vector_Sz):
    with open(preprocessed_path, 'r') as f:
        csv_reader = csv.reader(f)

        batch_size = 10000
        i = 0
        docs1 = []
        docs2 = []
        idxs = []
        labels = []
        # for each sample
        for l in tqdm(csv_reader, total=vector_Sz):

            labels.append(l[0])
            docs1.append(json.loads(l[1]))
            docs2.append(json.loads(l[2]))
            idxs.append(ordered_idxs[i])
            i += 1
            if len(labels) >= batch_size:
                x1 = scaler.transform(transformer.transform(docs1).todense())
                x2 = scaler.transform(transformer.transform(docs2).todense())
                XX[idxs, :] = secondary_scaler.transform(np.abs(x1 - x2))
                Y[idxs] = labels

                docs1 = []
                docs2 = []
                idxs = []
                labels = []

        x1 = scaler.transform(transformer.transform(docs1).todense())
        x2 = scaler.transform(transformer.transform(docs2).todense())
        XX[idxs, :] = secondary_scaler.transform(np.abs(x1 - x2))
        Y[idxs] = labels
        XX.flush()
        Y.flush()


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def main(args):

    train_sz, test_sz = 0, 0

    train_raw_texts = []
    with open(args.train_path, 'r') as f:
        reader = csv.reader(f)
        for _, ppt0, ppt1 in reader:
            train_raw_texts.append(json.loads(ppt0))
            train_raw_texts.append(json.loads(ppt1))
            train_sz += 1

    with open(args.test_path, 'r') as f:
        for _ in f:
            test_sz += 1


    print('Fitting transformer...', flush=True)
    start_time = time.time()
    transformer, scaler, secondary_scaler = fit_transformers(train_raw_texts, char_n=args.char_n)
    print(f'took {(time.time() - start_time)/60} seconds')
    feature_sz = len(transformer.get_feature_names())

    del train_raw_texts

    print('Vectorizing train set...', flush=True)
    print(f'tyring to allocate array of shape: {(train_sz, feature_sz)}')
    XX_train = np.memmap(TEMP_DATA_PATH + 'vectorized_XX_train.npy', dtype='float32', mode='w+', shape=(train_sz, feature_sz))
    Y_train = np.memmap(TEMP_DATA_PATH + 'Y_train.npy', dtype='int32', mode='w+', shape=(train_sz,))
    train_idxs = np.array(range(train_sz))
    np.random.shuffle(train_idxs)

    vectorize(
        XX_train,
        Y_train,
        train_idxs,
        transformer,
        scaler,
        secondary_scaler,
        args.train_path,
        train_sz
    )

    print('Vectorizing test set...', flush=True)
    XX_test = np.memmap(TEMP_DATA_PATH + 'vectorized_XX_test.npy', dtype='float32', mode='w+', shape=(test_sz, feature_sz))
    Y_test = np.memmap(TEMP_DATA_PATH + 'Y_test.npy', dtype='int32', mode='w+', shape=(test_sz,))
    test_idxs = np.array(range(test_sz))
    np.random.shuffle(test_idxs)

    vectorize(
        XX_test,
        Y_test,
        test_idxs,
        transformer,
        scaler,
        secondary_scaler,
        args.test_path,
        test_sz
    )

    print('Tuning parameters...', flush=True)

    param_dist = {'alpha': loguniform(1e-4, 1e0)}
    batch_size = 100
    clf = SGDClassifier(loss='log', alpha=0.01, n_jobs=32)
    n_iter_search = args.num_search_iters
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, verbose=2)
    for idxs in chunker(range(train_sz), batch_size):
            random_search.fit(XX_train[idxs, :], Y_train[idxs])
            break

    print('Best params:', random_search.best_params_)

    print('Training classifier...', flush=True)
    clf = SGDClassifier(loss='log', alpha=random_search.best_params_['alpha'], n_jobs=32)
    # batch_size = 50000
    batch_size = 128
    num_epochs = 200
    aucs = []
    for i in trange(num_epochs):
        print('Epoch - ', i)
        print('-' * 30)
        for idxs in chunker(range(train_sz), batch_size):
            clf.partial_fit(XX_train[idxs, :], Y_train[idxs], classes=[0, 1])

        probs = clf.predict_proba(XX_test)[:, 1]
        preds = clf.predict(XX_test)
        fpr, tpr, thresh = roc_curve(Y_test, probs)
        res = valla.utils.eval_metrics.av_metrics(Y_test, preds, probs)
        wandb.log(res)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print('AUC: ', roc_auc)
        wandb.log({'auc': roc_auc})
        with open(TEMP_DATA_PATH + 'experiment_data.p', 'wb') as f:
            pickle.dump((
                aucs,
                clf,
                roc_auc,
                transformer,
                scaler,
                secondary_scaler,
                feature_sz,
                train_sz,
                train_idxs,
                test_sz,
                test_idxs
            ), f)

    with open(TEMP_DATA_PATH + 'large_model.p', 'wb') as f:
        pickle.dump((clf, transformer, scaler, secondary_scaler), f)

    # from pan20_verif_evaluator import evaluate_all
    results = valla.utils.eval_metrics.evaluate_all(Y_test, probs)
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdHominem - Siamese Network for Authorship Verification')
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--num_search_iters', type=int, default=2)
    parser.add_argument('--char_n', type=int, default=3)

    args = parser.parse_args()
    with wandb.init(config=vars(args), project=args.project):
        main(args)
