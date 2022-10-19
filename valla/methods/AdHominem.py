"""
adapted from https://github.com/boenninghoff/AdHominem

We highly recommend using our reimpoementation of this, found in torched_AdHominem.py
"""
import tensorflow as tf
from valla.methods.ablations.AdHominem_core import AdHominem
import pickle
import os
import argparse
import wandb
import logging

logging.basicConfig(level=logging.DEBUG)


def sweep_adhominem(config=None, project=None, device='0', dataset='test', data_path='.'):
    with wandb.init(project=project, config=config):
        config = wandb.config
        os.environ["CUDA_VISIBLE_DEVICES"] = device

        config['vocab_size_tok'] = config['vocab_sizes'][0]
        config['vocab_size_char'] = config['vocab_sizes'][1]

        data_file = f'{dataset}_test_iid_fasttext_{config["vocab_size_tok"]}_{config["vocab_size_char"]}_30_300.pkl'
        pth = os.path.join(data_path, data_file)

        logging.info(f'loading data from {pth}')

        with open(pth, 'rb') as f:
            docs_L_tr, docs_R_tr, labels_tr, \
            docs_L_te, docs_R_te, labels_te, \
            V_w, E_w, V_c = pickle.load(f)

        config['t_s'] = 0.91
        config['t_d'] = 0.09
        config['loss'] = 'modified_contrastive'

        config['N_tr'] = len(labels_tr)
        config['N_dev'] = len(labels_te)
        config['len_V_c'] = len(V_c)
        config['len_V_w'] = len(V_w)

        config['model'] = 'AdHominem'
        wandb.config.update(config)

        # add vocabularies to dictionary
        config['V_w'] = V_w
        config['V_c'] = V_c

        logging.info('initializing AdHominem model')

        adhominem = AdHominem(hyper_parameters=config,
                              E_w_init=E_w)

        # start training
        train_set = (docs_L_tr, docs_R_tr, labels_tr)
        test_set = (docs_L_te, docs_R_te, labels_te)

        logging.info('launching training')

        adhominem.train_model(train_set, test_set)

        # close session
        adhominem.sess.close()



def main():

    parser = argparse.ArgumentParser(description='AdHominem - Siamese Network for Authorship Verification')
    parser.add_argument('--D_c', default=10, type=int)  # character embedding dimension
    parser.add_argument('--D_r', default=30, type=int)  # character representation dimension
    parser.add_argument('--w', default=4, type=int)  # length of 1D-CNN sliding window
    parser.add_argument('--D_w', default=300, type=int)  # dimension of word embeddings
    parser.add_argument('--D_s', default=50, type=int)  # dimension sentence embeddings
    parser.add_argument('--D_d', default=50, type=int)  # dimension of document embedding
    parser.add_argument('--D_mlp', default=60, type=int)  # final output dimension
    parser.add_argument('--T_c', default=15, type=int)  # maximum number of characters per words
    parser.add_argument('--T_w', default=20, type=int)  # maximum number of words per sentence
    parser.add_argument('--T_s', default=50, type=int)  # maximum number of sentences per document
    parser.add_argument('--t_s', default=0.91, type=float)  # boundary for similar pairs (close to one)
    parser.add_argument('--t_d', default=0.09, type=float)  # boundary for dissimilar pairs (close to zero)
    parser.add_argument('--epochs', default=100, type=int)  # total number of epochs
    parser.add_argument('--train_word_embeddings', default=False, type=bool)  # fine-tune pre-trained word embeddings
    parser.add_argument('--batch_size_tr', default=32, type=int)  # batch size for training
    parser.add_argument('--batch_size_te', default=128, type=int)  # batch size for evaluation
    parser.add_argument('--initial_learning_rate', default=0.002, type=float)  # initial learning rate
    parser.add_argument('--keep_prob_cnn', default=0.8, type=float)  # dropout for 1D-CNN layer
    parser.add_argument('--keep_prob_lstm', default=0.9, type=float)  # variational dropout for BiLSTM layer
    parser.add_argument('--keep_prob_att', default=0.9, type=float)  # dropout for attention layer
    parser.add_argument('--keep_prob_metric', default=0.8, type=float)  # dropout for metric learning layer
    parser.add_argument('--results_file', default='results.txt', type=str)
    parser.add_argument('--loss', default='modified_contrastive', type=str)
    parser.add_argument('--no_cnn', action='store_true')
    parser.add_argument('--no_fasttext', action='store_true')
    parser.add_argument('--vocab_wordemb_file', default='data/amazon.pkl', type=str)
    # /home/jtyo/Repos/AuthorshipAttribution/data/_gutenburg/train_test_adhominem.pkl
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--flatten', action='store_true')
    parser.add_argument('--proj_name', type=str)
    parser.add_argument('--vocab_size_tok', type=int, default=5000)
    parser.add_argument('--vocab_size_char', type=int, default=250)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--save_model_path', type=str, default=None)

    hyper_parameters = vars(parser.parse_args())

    os.environ["CUDA_VISIBLE_DEVICES"] = hyper_parameters['device']

    wandb.login()
    wandb_project = hyper_parameters['proj_name']
    hyper_parameters['model'] = 'AdHominem'

    # load docs, vocabularies and initialized word embeddings
    with open(hyper_parameters['vocab_wordemb_file'], 'rb') as f:
        docs_L_tr, docs_R_tr, labels_tr, \
        docs_L_te, docs_R_te, labels_te, \
        V_w, E_w, V_c = pickle.load(f)

    hyper_parameters['N_tr'] = len(labels_tr)
    hyper_parameters['N_dev'] = len(labels_te)
    hyper_parameters['len_V_c'] = len(V_c)
    hyper_parameters['len_V_w'] = len(V_w)

    hyper_parameters['model'] = 'AdHominem'

    wandb.init(project=wandb_project, config=hyper_parameters)

    # add vocabularies to dictionary
    hyper_parameters['V_w'] = V_w
    hyper_parameters['V_c'] = V_c

    if hyper_parameters['load_model_path'] is not None:
        # load trained model and hyper-parameters
        with open(os.path.join(hyper_parameters['load_model_path']), 'rb') as f:
            parameters = pickle.load(f)

        # overwrite old variables
        for hp in hyper_parameters:
            parameters["hyper_parameters"][hp] = hyper_parameters[hp]

        adhominem = AdHominem(hyper_parameters=parameters["hyper_parameters"],
                              theta_init=parameters['theta'],
                              E_w_init=parameters['theta_E'])
                              # E_w_init=E_w)

    else:

        adhominem = AdHominem(hyper_parameters=hyper_parameters,
                              E_w_init=E_w)

    # start training
    train_set = (docs_L_tr, docs_R_tr, labels_tr)
    test_set = (docs_L_te, docs_R_te, labels_te)

    adhominem.train_model(train_set, test_set)

    # close session
    adhominem.sess.close()


if __name__ == '__main__':
    main()
