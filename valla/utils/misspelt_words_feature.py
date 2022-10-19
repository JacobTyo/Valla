import os
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.probability import FreqDist
import pickle
import numpy as np


def misspelled_arr(misspelled_text_file_path):
    """
        Given a texfile of mispelled words,
        return an array of misspellings.
    """
    f = open(misspelled_text_file_path, "r")
    Lines = f.readlines()

    mis_arr_temp = []

    for t in Lines:
        w = t.split('//')[1].strip()
        w_arr = w.split(',')
        mis_arr_temp.extend(w_arr)

    mis_arr = [w.strip().lower() for w in mis_arr_temp]

    return mis_arr


def common_typos(typos_text_file_path):
    """
        Given a textfile of typos,
        return an array of common misspellings.

        Reference: https://www.lexico.com/grammar/common-misspellings
    """
    f = open(typos_text_file_path)

    Lines = f.readlines()

    typos = [t.split()[-1].lower() for t in Lines]

    return typos


def brit_spelling(file_path):
    """
        Given a text file of British typos,
        return an array of British spellings of words.
    """
    # https://www.lexico.com/grammar/british-and-spelling

    f = open(file_path)
    Lines = f.readlines()

    b = []

    for l in Lines:
        w = l.split('\t')[0].lower()
        word = w.split()
        for x in word:
            if x != '\n':
                b.append(x)

    return b


def determiner(file_path):
    """
        Given a file of typos with determiners,
        return an array of mistyped determiners.
    """
    f = open(file_path)

    L = f.readlines()

    d = [w.split()[0].lower() for w in L]

    return d


def create_misspellings_dict(data_dir):
    dictionary = {
        'typos': set(common_typos(os.path.join(data_dir, 'common_typos.txt'))),
        'common': set(misspelled_arr(os.path.join(data_dir, 'mis_words.txt'))),
        'british': set(brit_spelling(os.path.join(data_dir, 'brit_spelling.txt'))),
        'determiner': set(determiner(os.path.join(data_dir, 'determiner_err.txt')))
    }
    return dictionary


class MisspellingsFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, data_dir):
        dict_path = os.path.join(data_dir, 'misspellings_dict.p')
        if os.path.exists(dict_path):
            with open(dict_path, 'rb') as f:
                self.misspelling_dict = pickle.load(f)
        else:
            self.misspelling_dict = create_misspellings_dict(data_dir)
            with open(dict_path, 'wb') as f:
                pickle.dump(self.misspelling_dict, f)

    def fit(self, x, y=None):
        return self

    def _process(self, x):
        fdist = FreqDist(x['tokens'])
        result = []
        for k in self.misspelling_dict.keys():
            result.append(sum([fdist[w] for w in self.misspelling_dict[k]]) / len(x['tokens']))
        return result

    def transform(self, X):
        return list(map(self._process, X))

    def get_feature_names(self):
        return list(self.misspelling_dict.keys())