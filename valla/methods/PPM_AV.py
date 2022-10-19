# -*- coding: utf-8 -*-

"""
Adapted from: https://github.com/pan-webis-de/pan-code/tree/master/clef21/authorship-verification

 A baseline authorship verificaion method based on text compression.
 Given two texts text1 and text2 it calculates the cross-entropy of text2 using the Prediction by Partical Matching (PPM) compression model of text1 and vice-versa.
 Then, the mean and absolute difference of the two cross-entropies are used to estimate a score in [0,1] indicating the probability the two texts are written by the same author.
 The prediction model is based on logistic regression and can be trained using a collection of training cases (pairs of texts by the same or different authors).
 Since the verification cases with a score exactly equal to 0.5 are considered to be left unanswered, a radius around this value is used to determine what range of scores will correspond to the predetermined value of 0.5.

 The method is based on the following paper:
     William J. Teahan and David J. Harper. Using compression-based language models for text categorization. In Language Modeling and Information Retrieval, pp. 141-165, 2003
 The current implementation is based on the code developed in the framework of a reproducibility study:
     M. Potthast, et al. Who Wrote the Web? Revisiting Influential Author Identification Research Applicable to Information Retrieval. In Proc. of the 38th European Conference on IR Research (ECIR 16), March 2016.
     https://github.com/pan-webis-de/teahan03
 Questions/comments: stamatatos@aegean.gr

 It can be applied to datasets of PAN-21 cross-domain authorship verification task.
 See details here: http://pan.webis.de/clef21/pan21-web/author-identification.html
 Dependencies:
 - Python 2.7 or 3.6 (we recommend the Anaconda Python distribution)

 Usage from command line:
    > python pan21-authorship-verification-baseline-compressor.py -i EVALUATION-FILE -o OUTPUT-DIRECTORY [-m MODEL-FILE]
 EVALUATION-DIRECTORY (str) is the full path name to a PAN-20 collection of verification cases (each case is a pair of texts)
 OUTPUT-DIRECTORY (str) is an existing folder where the predictions are saved in the PAN-20 format
 Optional parameter:
     MODEL-FILE (str) is the full path name to the trained model (default=model_small.joblib, a model already trained on the small training dataset released by PAN-20 using logistic regression with PPM order = 5)
     RADIUS (float) is the radius around the threshold 0.5 to leave verification cases unanswered (dedault = 0.05). All cases with a value in [0.5-RADIUS, 0.5+RADIUS] are left unanswered.

 Example:
     > python pan21-authorship-verification-baseline-compressor.py -i "mydata/pan20-input" -o "mydata/pan20-answers" -m "mydata/model_small.joblib"

 Additional functions (train_data and train_model) are provided to prepare training data and train a new model.

 Supplementary files:
    data-small.txt: training data extracted from the small dataset provided by PAN-20 authorship verification task
    model.joblib: trained model using logistic regression, PPM order=5, using data of data-small.txt
"""

from __future__ import print_function

import pickle
from math import log
import os
import json
import time
import argparse
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from typing import Dict
import logging
from valla.utils.eval_metrics import av_metrics
from valla.dsets.loaders import get_av_dataset, get_aa_dataset
from valla.utils.dataset_utils import list_dset_to_dict
import wandb
from multiprocessing import Pool



class Model(object):
    # cnt - count of characters read
    # modelOrder - order of the model
    # orders - List of Order-Objects
    # alphSize - size of the alphabet
    def __init__(self, order, alphSize):
        self.cnt = 0
        self.alphSize = alphSize
        self.modelOrder = order
        self.orders = []
        for i in range(order + 1):
            self.orders.append(Order(i))

    # print the model
    # TODO: Output becomes too long, reordering on the screen has to be made
    def printModel(self):
        s = "Total characters read: " + str(self.cnt) + "\n"
        for i in range(self.modelOrder + 1):
            self.printOrder(i)

    # print a specific order of the model
    # TODO: Output becomes too long, reordering on the screen has to be made
    def printOrder(self, n):
        o = self.orders[n]
        s = "Order " + str(n) + ": (" + str(o.cnt) + ")\n"
        for cont in o.contexts:
            if n > 0:
                s += "  '" + cont + "': (" + str(o.contexts[cont].cnt) + ")\n"
            for char in o.contexts[cont].chars:
                s += "     '" + char + "': " + \
                     str(o.contexts[cont].chars[char]) + "\n"
        s += "\n"
        print(s)

    # updates the model with a character c in context cont
    def update(self, c, cont):
        if len(cont) > self.modelOrder:
            raise NameError("Context is longer than model order!")

        order = self.orders[len(cont)]
        if not order.hasContext(cont):
            order.addContext(cont)
        context = order.contexts[cont]
        if not context.hasChar(c):
            context.addChar(c)
        context.incCharCount(c)
        order.cnt += 1
        if order.n > 0:
            self.update(c, cont[1:])
        else:
            self.cnt += 1

    # updates the model with a string
    def read(self, s):
        if len(s) == 0:
            return
        for i in range(len(s)):
            cont = ""
            if i != 0 and i - self.modelOrder <= 0:
                cont = s[0:i]
            else:
                cont = s[i - self.modelOrder:i]
            self.update(s[i], cont)

    # return the models probability of character c in content cont
    def p(self, c, cont):
        if len(cont) > self.modelOrder:
            raise NameError("Context is longer than order!")

        order = self.orders[len(cont)]
        if not order.hasContext(cont):
            if order.n == 0:
                return 1.0 / self.alphSize
            return self.p(c, cont[1:])

        context = order.contexts[cont]
        if not context.hasChar(c):
            if order.n == 0:
                return 1.0 / self.alphSize
            return self.p(c, cont[1:])
        return float(context.getCharCount(c)) / context.cnt

    # merge this model with another model m, esentially the values for every
    # character in every context are added
    def merge(self, m):
        if self.modelOrder != m.modelOrder:
            raise NameError("Models must have the same order to be merged")
        if self.alphSize != m.alphSize:
            raise NameError("Models must have the same alphabet to be merged")
        self.cnt += m.cnt
        for i in range(self.modelOrder + 1):
            self.orders[i].merge(m.orders[i])

    # make this model the negation of another model m, presuming that this
    # model was made my merging all models
    def negate(self, m):
        if self.modelOrder != m.modelOrder or self.alphSize != m.alphSize or self.cnt < m.cnt:
            raise NameError("Model does not contain the Model to be negated")
        self.cnt -= m.cnt
        for i in range(self.modelOrder + 1):
            self.orders[i].negate(m.orders[i])


class Order(object):
    # n - whicht order
    # cnt - character count of this order
    # contexts - Dictionary of contexts in this order
    def __init__(self, n):
        self.n = n
        self.cnt = 0
        self.contexts = {}

    def hasContext(self, context):
        return context in self.contexts

    def addContext(self, context):
        self.contexts[context] = Context()

    def merge(self, o):
        self.cnt += o.cnt
        for c in o.contexts:
            if not self.hasContext(c):
                self.contexts[c] = o.contexts[c]
            else:
                self.contexts[c].merge(o.contexts[c])

    def negate(self, o):
        if self.cnt < o.cnt:
            raise NameError(
                "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
        self.cnt -= o.cnt
        for c in o.contexts:
            if not self.hasContext(c):
                raise NameError(
                    "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
            else:
                self.contexts[c].negate(o.contexts[c])
        empty = [c for c in self.contexts if len(self.contexts[c].chars) == 0]
        for c in empty:
            del self.contexts[c]


class Context(object):
    # chars - Dictionary containing character counts of the given context
    # cnt - character count of this context
    def __init__(self):
        self.chars = {}
        self.cnt = 0

    def hasChar(self, c):
        return c in self.chars

    def addChar(self, c):
        self.chars[c] = 0

    def incCharCount(self, c):
        self.cnt += 1
        self.chars[c] += 1

    def getCharCount(self, c):
        return self.chars[c]

    def merge(self, cont):
        self.cnt += cont.cnt
        for c in cont.chars:
            if not self.hasChar(c):
                self.chars[c] = cont.chars[c]
            else:
                self.chars[c] += cont.chars[c]

    def negate(self, cont):
        if self.cnt < cont.cnt:
            raise NameError(
                "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
        self.cnt -= cont.cnt
        for c in cont.chars:
            if (not self.hasChar(c)) or (self.chars[c] < cont.chars[c]):
                raise NameError(
                    "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
            else:
                self.chars[c] -= cont.chars[c]
        empty = [c for c in self.chars if self.chars[c] == 0]
        for c in empty:
            del self.chars[c]


# calculates the cross-entropy of the string 's' using model 'm'
def h(m, s):
    n = len(s)
    h = 0
    for i in range(n):
        if i == 0:
            context = ""
        elif i <= m.modelOrder:
            context = s[0:i]
        else:
            context = s[i - m.modelOrder:i]
        h -= log(m.p(s[i], context), 2)
    return h / n


# Calculates the cross-entropy of text2 using the model of text1 and vice-versa
# Returns the mean and the absolute difference of the two cross-entropies
def distance(text1, text2, ppm_order=5, label=None):
    mod1 = Model(ppm_order, 256)
    mod1.read(text1)
    d1 = h(mod1, text2)  # this is essentially perplexity of model trained on 1 and tested on 2
    mod2 = Model(ppm_order, 256)
    mod2.read(text2)
    d2 = h(mod2, text1)
    # peroplexity 1-on-2, perplexity 2-on-1, average perplexity, perplexity difference
    if label is not None:
        return [label, [d1, d2, (d1 + d2) / 2.0, abs(d1 - d2)]]
    return [d1, d2, (d1 + d2) / 2.0, abs(d1 - d2)]


class ppm_training_data_generator:
    def __init__(self, train_data, sample_passes=1, ppm_order=5, dset_name='test', cache_dir='test', num_workers=0,
                 reprocess=False):
        self.train_data = train_data
        self.sample_passes = sample_passes
        self.ppm_order = ppm_order
        self.dset_name = dset_name
        self.cache_dir = cache_dir
        self.num_workers = num_workers
        self.reprocess = reprocess

        self.classifier_training_data = []

    def update_classifier_training_data(self, sample_point):
        self.classifier_training_data.append(sample_point)

    def get_train_data(self):
        # assume an AA training set for this I rekon
        assert isinstance(self.train_data, Dict), 'we want the dictionary version of the dataset for trainnig'

        cache_path = os.path.join(self.cache_dir, f'{self.dset_name}_{self.ppm_order}_{self.sample_passes}.data')

        if os.path.exists(cache_path) and not self.reprocess:
            # get the cached training data
            logging.info(f'Gather PPM Training Data {self.dset_name}: getting training data from {cache_path}')
            with open(cache_path, 'r') as f:
                self.classifier_training_data = json.load(f)
        else:
            with Pool(processes=self.num_workers) as pool:

                logging.info(
                    f'Gather PPM Training Data {self.dset_name}: getting training data for the PPM_AV method on the '
                    f'{self.dset_name} dataset, using {self.num_workers} workers')
                async_results, idx = {}, 0
                for _ in range(self.sample_passes):
                    for auth_id, texts in tqdm(self.train_data.items(), 'Step'):
                        for text in texts:
                            # one same and one
                            text = text[:50000]
                            same_text = random.choice(self.train_data[auth_id])[:50000]
                            diff_text = random.choice(self.train_data[random.choice(list(self.train_data.keys()))])[:50000]

                            async_results[idx] = pool.apply_async(distance, (text, same_text, self.ppm_order, 1))
                            idx += 1
                            async_results[idx] = pool.apply_async(distance, (text, diff_text, self.ppm_order, 0))
                            idx += 1

                logging.info(f'Gather PPM Training Data {self.dset_name}: finished launching, now awaiting the processing')
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
                            self.update_classifier_training_data(res)
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
                            f'Gather PPM Training Data {self.dset_name}: {res_left} results remaining in queue, {elapsed:.2f}, {num_removed} removed, {loops} loops ran.')
                        logging.info(
                            f'Gather PPM Training Data {self.dset_name}: approximately {(res_left / (num_removed / 30)) / 60} minutes remaining')
                        last_check = time.time()
                        num_removed = 0

                logging.info(f'Gather PPM Training Data {self.dset_name}: all {self.dset_name} data gathered')
                # now save the training data
                logging.info(f'caching data for training the classifier to {cache_path}')
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(self.train_data, f)
        return self.classifier_training_data


def train_classifier(X_train, y_train):
    logging.info('fitting the logistic regression model')
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    return logreg


# Trains the logistic regression model
def train_and_eval_model(train_data, test_data, sample_passes=1, ppm_order=5, dset_name='test', cache_dir=None,
                         num_workers=0, reprocess=False):
    data_generator = ppm_training_data_generator(train_data, sample_passes, ppm_order, dset_name, cache_dir,
                                                 num_workers, reprocess)
    classifier_training_data = data_generator.get_train_data()
    y_train = []
    X_train = []
    for lbl, dists in classifier_training_data:
        y_train.append(lbl)
        X_train.append(dists)
    # now train the logistic regression classifier
    logreg = train_classifier(X_train, y_train)

    # just pickle the log reg for now
    if cache_dir is not None:
        logreg_save_file = os.path.join(cache_dir, f'logreg_{dset_name}_{ppm_order}_{sample_passes}.clf')
        logging.info(f'saving the fit logistic regression classifier to {logreg_save_file}')
        os.makedirs(os.path.dirname(logreg_save_file), exist_ok=True)
        with open(logreg_save_file, 'wb') as f:
            pickle.dump(logreg, f)

    # now we can just evaluate
    evaluate_model(test_data, logreg, ppm_order, num_workers)


def eval_sample(txt0, txt1, ppm_order, logreg, true_lbl=None):
    dists = distance(txt0, txt1, ppm_order)
    proba = logreg.predict_proba([dists])
    if true_lbl is not None:
        return proba[0][1], true_lbl
    return proba[0][1]


def evaluate_model(test_data, logreg, ppm_order, num_workers):
    probas_and_true_lbls = []

    with Pool(processes=num_workers) as pool:
        logging.info(
            f'PPM_AV: evaluating')
        async_results, idx = {}, 0
        for true_lbl, txt0, txt1 in tqdm(test_data, 'Evaluating'):
            async_results[idx] = pool.apply_async(eval_sample, (txt0, txt1, ppm_order, logreg, true_lbl))
            idx += 1

        logging.info(f'PPM_AV: finished launching, now awaiting the processing')
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
                    probas_and_true_lbls.append(res)
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
                    f'PPM_AV: {res_left} results remaining in queue, {elapsed:.2f}, {num_removed} removed, '
                    f'{loops} loops ran.')
                logging.info(
                    f'PPM_AV: approximately {(res_left / (num_removed / 30)) / 60} minutes remaining')
                last_check = time.time()
                num_removed = 0

        logging.info(f'PPM_AV: all evaluation results gathered')
        probas, true_lbls = [], []
        for proba, lbl in probas_and_true_lbls:
            probas.append(proba)
            true_lbls.append(lbl)
    # now calculate metrics and call it good
    results = av_metrics(true_lbls, probas=probas, threshold=0.5)
    logging.info('finished, results:')
    for k, v in results.items():
        logging.info(f'\t{k}: {v:.4f}')
    wandb.log(results)


def main():
    parser = argparse.ArgumentParser(
        description='PAN-21 Cross-domain Authorship Verification task: Baseline Compressor')
    parser.add_argument('--train_path', type=str, help='Path to the folder containing the pairs.jsonl')
    parser.add_argument('--test_path', type=str, help='Path to an output folder')
    parser.add_argument('--cache_path', type=str, default='test')
    parser.add_argument('--wandb_project', type=str, default='test')
    parser.add_argument('--dset_name', type=str, default='test')
    parser.add_argument('--ppm_order', type=int, default=5)
    parser.add_argument('--sample_passes', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--reprocess', action='store_true')
    args = parser.parse_args()

    wandb_args = vars(args)
    wandb_args['model'] = 'PPM_AV'

    # get the training and test data

    with wandb.init(project=args.wandb_project, config=wandb_args):

        train_dataset = list_dset_to_dict(get_aa_dataset(args.train_path))
        test_dataset = get_av_dataset(args.test_path)

        train_and_eval_model(train_dataset,
                             test_dataset,
                             sample_passes=args.sample_passes,
                             ppm_order=5,
                             dset_name=args.dset_name,
                             cache_dir=args.cache_path,
                             num_workers=args.num_workers,
                             reprocess=args.reprocess)


if __name__ == '__main__':
    main()
