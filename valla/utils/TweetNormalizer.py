"""
pulled from https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
"""
from emoji import demojize
from nltk.tokenize import TweetTokenizer
from valla.dsets.loaders import get_aa_dataset
from valla.utils.dataset_utils import write_aa_dataset, list_dset_to_dict
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)

tokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())


if __name__ == "__main__":
    # use this to normalize a dataset w.r.t. this tweet preprocessing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    args = parser.parse_args()

    logging.info(f'getting dataset from {args.dataset_path}')
    dset = get_aa_dataset(args.dataset_path)

    logging.info('normalizing tweets')
    # this is a list of examples [label, text]
    # just process texts and write - lazy but ez, efficency not too important here
    for i in tqdm(range(len(dset))):
        dset[i] = [dset[i][0], normalizeTweet(dset[i][1])]

    logging.info('saving dataset with .tweet filename extension')
    # now write the processed dset
    write_aa_dataset(list_dset_to_dict(dset), file_path=args.dataset_path+'.tweet')
