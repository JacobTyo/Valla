import csv
import os
import pickle
import json
import glob
from tqdm.auto import trange, tqdm
import sys
import argparse
import nltk
from nltk.corpus import conll2000
from nltk.tag.perceptron import PerceptronTagger
from valla.dsets.loaders import get_av_dataset

import re
import numpy as np

import valla.dsets.loaders

'''
DATA_DIR = '/media/disk1/social/troll_tweets/data/pan_clustering/pan20_large/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
DATA_PATH = DATA_DIR + 'pan20-authorship-verification-training-large.jsonl'
TEMP_DATA_PATH = 'temp_data/large_model_training_data/'
DATA_DIR = '/scratch/jnw301/av/data/pan20_large/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
DATA_PATH = DATA_DIR + 'pan20-authorship-verification-training-large.jsonl'
TEMP_DATA_PATH = '/scratch/jnw301/pan2021_av/temp_data/large_model_training_data/'
'''

dirname = os.path.dirname(__file__)
regex_chunker = None
ml_chunker = None
tnlp_regex_chunker = None

DATA_DIR = '/scratch/jnw301/av/data/pan20_large/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
DATA_PATH = DATA_DIR + 'pan20-authorship-verification-training-large.jsonl'
TEMP_DATA_PATH = '/scratch/jnw301/pan2021_av/temp_data/large_model_training_data/'

tagger = nltk.data.load(os.path.join(dirname, "pos_tagger/treebank_brill_aubt.pickle"))
perceptron_tagger = PerceptronTagger()
treebank_tokenizer = nltk.tokenize.TreebankWordTokenizer()

nlp_stanza = None  # stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)
nlp_spacy = None  # spacy.load("en_core_web_sm", disable=['ner'])

NUM_MACHINES = 10

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


def preprocess_text(text):
    # remove URLs
    text = text.lower()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' URL ', text)

    return text


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


def pos_tag_chunk(pos_tags, chunker):
    parse_tree = chunker.parse(pos_tags)
    subtree_expansions = []
    for subt in parse_tree:
        extract_subtree_expansions(subt, subtree_expansions)
    return list(map(chunk_to_str, parse_tree)), subtree_expansions


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


def get_nltk_pos_tag_based_regex_chunker():
    global regex_chunker
    if regex_chunker is not None:
        return regex_chunker
    regex_chunker = nltk.RegexpParser(grammar)
    return regex_chunker


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


def train_chunker():
    train_sents = conll2000.chunked_sents('train.txt')
    test_sents = conll2000.chunked_sents('test.txt')
    chunker = ConsecutiveNPChunker(train_sents)
    print(chunker.evaluate(test_sents))
    pickle.dump(chunker, open(os.path.join(dirname, 'temp_data/chunker.p'), 'wb'))


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AdHominem - preprocessing')

    # self, test_split = 0.2, T_w = 20, D_w = 300, vocab_size_token = 15000, vocab_size_chr = 125, dataset = 'amazon',
    # train_path = None, test_path = None, embeddings = 'fasttext'

    parser.add_argument('--instance_id', type=int)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--mode', type=str, default='fast', help='other option: accurate')
    parser.add_argument('--just_test', action='store_true')

    args = parser.parse_args()

    # instance_id = int(sys.argv[1])
    instance_id = args.instance_id
    # train_path = sys.argv[2]
    # test_path = sys.argv[3]
    train_path = args.train_path
    test_path = args.test_path
    print('Instance ID for this machine:', instance_id, flush=True)

    train_samples = get_av_dataset(train_path)
    test_samples = get_av_dataset(test_path)

    val_test = 'test' if 'test' in test_path else 'val'
    # train_ids, test_ids, _, _ = pickle.load(open(TEMP_DATA_PATH + 'dataset_partition.p', 'rb'))

    # total_recs = len(train_ids) + len(test_ids)
    total_recs = len(train_samples)
    job_sz = total_recs // NUM_MACHINES
    start_rec = instance_id * job_sz
    end_rec = (instance_id + 1) * job_sz

    train_save_file = os.path.join(args.save_path, f'feat_diff_train_{args.mode}.csv')
    test_save_file = os.path.join(args.save_path, f'feat_diff_{val_test}_{args.mode}.csv')

    if not args.just_test:
        print('Recs on this machine:', (end_rec - start_rec), flush=True)
        i = 0
        with open(train_save_file, 'a') as f_train:
            train_writer = csv.writer(f_train)
            for label, text0, text1 in tqdm(train_samples, desc='preprocessing training set'):
                #for l in tqdm(f, total=total_recs):
                i += 1
                if i < start_rec or i > end_rec:
                    continue
                # d = json.loads(l)
                preprocessed = [
                    label,
                    json.dumps(prepare_entry(text0, mode=args.mode, tokenizer='casual')),
                    json.dumps(prepare_entry(text1, mode=args.mode, tokenizer='casual'))
                ]
                train_writer.writerow(preprocessed)

    if instance_id == 0 or args.just_test:
        with open(test_save_file, 'a') as f_test:
            test_writer = csv.writer(f_test)
            for label, text0, text1 in tqdm(test_samples, desc='preprocessing testing set'):
                #for l in tqdm(f, total=total_recs):
                # i += 1
                # if i < start_rec or i > end_rec:
                #     continue
                # d = json.loads(l)
                preprocessed = [
                    label,
                    json.dumps(prepare_entry(text0, mode=args.mode, tokenizer='casual')),
                    json.dumps(prepare_entry(text1, mode=args.mode, tokenizer='casual'))
                ]
                test_writer.writerow(preprocessed)
