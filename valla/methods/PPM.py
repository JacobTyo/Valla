"""
This is a modified version of the python file found here:
    https://github.com/pan-webis-de/teahan03/blob/master/teahan03.py
"""
from math import log
import pickle
import os
import argparse
import logging
import wandb
import pandas as pd
from tqdm import tqdm
import time
from valla.dsets.loaders import aa_as_pandas, get_aa_dataset
from valla.utils.eval_metrics import aa_metrics

logging.basicConfig(level=logging.DEBUG)


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
            if(n > 0):
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
        if (order.n > 0):
            self.update(c, cont[1:])
        else:
            self.cnt += 1

    # updates the model with a string
    def read(self, s):
        if (len(s) == 0):
            return
        for i in range(len(s)):
            cont = ""
            if (i != 0 and i - self.modelOrder <= 0):
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
            if (order.n == 0):
                return 1.0 / self.alphSize
            return self.p(c, cont[1:])

        context = order.contexts[cont]
        if not context.hasChar(c):
            if (order.n == 0):
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


# returns model object loaded from 'mpath' using pickle
def load_model(mpath):
    f = open(mpath, "rb")
    m = pickle.load(f)
    f.close()
    return m


# stores model object 'model' to 'mpath' using pickle
def store_model(model, mpath):
    f = open(mpath, "wb")
    pickle.dump(model, f)
    f.close()


# calculates the cross-entropy of the string 's' using model 'm'
def h(m, s):
    n = len(s)
    _h = 0
    for i in range(n):
        if i == 0:
            context = ""
        elif i <= m.modelOrder:
            context = s[0:i]
        else:
            context = s[i - m.modelOrder:i]
        _h -= log(m.p(s[i], context), 2)
    return _h / n


# loads models of candidates in 'candidates' into 'models'
def load_models(pth):
    models = {}
    for file in os.listdir(pth):
        auth_id = file.split('_')[-1].split('.')[0]
        logging.info(f'loading model for author {auth_id}')
        models[auth_id] = load_model(os.path.join(pth, file))
    return models


# creates models of candidates in 'candidates'
# updates each model with any files stored in the subdirectory of 'corpusdir' named with the candidates name
# stores each model named under the candidates name in 'modeldir'
def create_models(data, order, alphsize):
    models = {}
    for i in data['labels'].unique():
        models[i] = Model(order, alphsize)
        logging.info(f"creating model for author {i}")

        for doc in tqdm(data['text'][data['labels'] == i]):
            models[i].read(doc)

    return models

# attributes the authorship, according to the cross-entropy ranking.
# attribution is saved in json-formatted structure 'answers'
def create_answers(test_data, models):
    logging.info("attributing authors to unknown texts")
    candidates = list(sorted(models.keys()))
    true_authors, authors, scores = [], [], []
    for true_author, text in tqdm(zip(test_data['labels'], test_data['text']), desc='test data'):
        hs = []
        for cand in candidates:
            hs.append(h(models[cand], text))
        m = min(hs)
        author = candidates[hs.index(m)]
        hs.sort()
        score = (hs[1] - m) / (hs[len(hs) - 1] - m)

        true_authors.append(true_author)
        authors.append(author)
        scores.append(score)

    return aa_metrics(true_authors, authors, scores, prefix='val/', no_auc=True)


def run_ppm(config=None, train_pth=None, test_pth=None):
    with wandb.init(config=config):
        config = wandb.config
        # get the datasets
        logging.info('getting the datasets')
        df_train = aa_as_pandas(get_aa_dataset(train_pth))
        df_eval = aa_as_pandas(get_aa_dataset(test_pth))

        models = create_models(df_train, config.order, config.alph_size)

        # create answers
        results = create_answers(df_eval, models)
        print(results)

        wandb.log(results)

# commandline argument parsing, calling the necessary methods
def main(params):

    wandb.login()

    wandb_project = params.experiment_name

    params.model = 'PPM'

    wandb.init(project=wandb_project, config=vars(params))

    # get the datasets
    logging.info('getting the datasets')
    df_train = aa_as_pandas(get_aa_dataset(params.train_dataset))
    df_eval = aa_as_pandas(get_aa_dataset(params.eval_dataset))
    # we don't need an eval set here (no hyperparameter tuning), so just use it as part of the training.
    df_train = pd.concat([df_train, df_eval], ignore_index=True)
    df_test = aa_as_pandas(get_aa_dataset(params.test_dataset))

    # something that loads models if they are cached?
    # if os.path.exists(params.save_path) and len(os.listdir(params.save_path)) > 0:
    #     logging.info('loading saved models')
    #     models = load_models(params.save_path)
    # else:
    models = create_models(df_train, params.order, params.alph_size)

    # create answers
    results = create_answers(df_test, models)

    # update the results keys

    wandb.log(results)

    # wandb.log(results, step=1)

    # save models
    logging.info(f'saving the {len(list(models.keys()))} models to {params.save_path}')
    if not os.path.exists(params.save_path):
        os.mkdir(params.save_path)
    for k in models.keys():
        save_pth = os.path.join(params.save_path, f'ppm_{k}.model')
        store_model(k, save_pth)
        wandb.save(save_pth)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reimplementation of  for AA (teahan03)")

    parser.add_argument('--experiment_name', type=str, default='imdb62',
                        help='the mlflow experiment name')
    parser.add_argument('--train_dataset', type=str,
                        default='/home/jtyo/data/Projects/On_the_SOTA_of_Authorship_Verification/datasets/imdb'
                                '/processed/imdb62/imdb62_train.csv')
    parser.add_argument('--eval_dataset', type=str,
                        default='/home/jtyo/data/Projects/On_the_SOTA_of_Authorship_Verification/datasets/imdb'
                                '/processed/imdb62/imdb62_AA_val.csv')
    parser.add_argument('--test_dataset', type=str,
                        default='/home/jtyo/data/Projects/On_the_SOTA_of_Authorship_Verification/datasets/imdb'
                                '/processed/imdb62/imdb62_AA_test.csv')
    parser.add_argument('--save_path', type=str, default='ppm_models')
    parser.add_argument('--order', type=int, default=5)
    parser.add_argument('--alph_size', type=int, default=256)

    args = parser.parse_args()

    main(args)

    # logging.warning('!!!!this is currently running the sweep function as a test!!!!!')
    # start = time.time()
    # run_ppm(args, args.train_dataset, args.eval_dataset)
    # logging.info(f'this run took {time.time() - start} seconds')
