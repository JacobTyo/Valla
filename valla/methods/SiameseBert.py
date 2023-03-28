import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from valla.utils.Evaluators import ContrastiveEvaluator, AaEvaluator
from sentence_transformers import models, SentenceTransformer
from sentence_transformers.readers import InputExample
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from sentence_transformers.losses.ContrastiveLoss import SiameseDistanceMetric
from sentence_transformers.losses import BatchAllTripletLoss, BatchHardTripletLoss
from valla.utils.Losses import ModifiedContrastiveLoss, MyContrastiveLoss
from valla.dsets.loaders import get_av_dataset, get_aa_dataset
from valla.utils.dataset_utils import list_dset_to_dict

import wandb
import argparse
import os
import numpy as np
import random
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)


class AADataset(Dataset):
    def __init__(self, data_path, true_hard_negatives=False):
        super(AADataset, self).__init__()
        self.data = list_dset_to_dict(get_aa_dataset(data_path))
        self.data_len = sum([len(x) for x in self.data.values()])
        # build a map to uniquely identify texts
        self.idx_to_txt_map = {}
        i = 0
        for auth, texts in self.data.items():
            for text_loc, _ in enumerate(texts):
                self.idx_to_txt_map[i] = {
                    'auth_id': auth,
                    'text_id': text_loc
                }
                i += 1
        self.author_list = list(self.data.keys())
        self.true_hard_negatives = true_hard_negatives
        self.hard_negative_data = []

    def __len__(self):
        if self.true_hard_negatives:
            return len(self.hard_negative_data)
        return self.data_len

    def __getitem__(self, item):
        if self.true_hard_negatives:
            return self.hard_negative_data[item]
        auth_id = self.idx_to_txt_map[item]['auth_id']
        txt_num = self.idx_to_txt_map[item]['text_id']
        text0 = self.data[auth_id][txt_num]
        if random.random() < 0.5:
            # different author sample
            label = 0
            auth2 = random.choice(self.author_list)
            while auth2 == auth_id:
                auth2 = random.choice(self.author_list)
            text1 = random.choice(self.data[auth2])
        else:
            # same author sample
            label = 1
            text1 = random.choice(self.data[auth_id])
        # now pick a random ~512 words from each text to send
        text0 = get_random_substring(text0)
        text1 = get_random_substring(text1)

        return InputExample(texts=[text0, text1], label=label)

    def listify_text(self, txt, chunk_len=512, max_txt_len=100000):
        txt = txt[:max_txt_len]
        chunked = []
        for i in range(0, len(txt), chunk_len):
            chunked.append(txt[i:i + chunk_len])
        return chunked

    def reset_hard_negatives(self, model, dist_metric):
        with torch.no_grad():
            logging.info('finding the hard positive and negatives.')
            embedded_texts = {}
            self.hard_negative_data = []
            for text_num, info in tqdm(self.idx_to_txt_map.items(), desc='encoding everything'):
                _this_t = self.data[info['auth_id']][info['text_id']]
                embedded_texts.setdefault(info['auth_id'], []).append(model.encode(get_random_substring(_this_t), show_progress_bar=False))

            for this_txt_idx, (auth, text_embeddings) in enumerate(tqdm(embedded_texts.items(), desc='finding hard negatives')):
                # TODO: I am not sure what is going on here actually -
                #   TODO: the dist_metric(text_embeddings, text_embeddings) compares all texts for the author,
                #    but I need one pos and neg for each text for each author
                same_samples = dist_metric(torch.tensor(np.array(text_embeddings)))
                for this_auth_txt_num, text_embedding in enumerate(text_embeddings):
                    # save the hardest positive
                    same_sample_idx = np.argmax(same_samples[this_auth_txt_num])
                    if same_sample_idx > this_auth_txt_num:
                        same_sample_idx -= 1
                    t1 = self.data[auth][this_auth_txt_num]
                    t2 = self.data[auth][same_sample_idx]
                    self.hard_negative_data.append(
                        InputExample(texts=[t1, t2], label=1))

                    # now get the highest
                    hardest_neg = [9999999999, [None, None]]
                    for other_auth_idx, (other_auth, other_text_embeddings) in enumerate(embedded_texts.items()):
                        if other_auth == auth:
                            continue
                        _text_embedding = torch.unsqueeze(torch.tensor(text_embedding), 0)
                        _other_text_embeddings = torch.tensor(np.array(other_text_embeddings))
                        diff_samples = dist_metric(torch.cat((_text_embedding, _other_text_embeddings), dim=0))[0][1:]
                        diff_sample_idx = np.argmin(diff_samples)
                        if diff_samples[diff_sample_idx] < hardest_neg[0]:
                            hardest_neg = [diff_samples[diff_sample_idx], [other_auth, diff_sample_idx]]
                    self.hard_negative_data.append(InputExample(texts=[self.data[auth][this_auth_txt_num], self.data[hardest_neg[1][0]][hardest_neg[1][1]]]))


class AAdataset_triplet(Dataset):
    def __init__(self, data_path, true_hard_negatives=False):
        super(AAdataset_triplet, self).__init__()
        self.data = list_dset_to_dict(get_aa_dataset(data_path))
        self.data_len = sum([len(x) for x in self.data.values()])
        # build a map to uniquely identify texts
        self.idx_to_txt_map = {}
        i = 0
        for auth, texts in self.data.items():
            for text_loc, _ in enumerate(texts):
                self.idx_to_txt_map[i] = {
                    'auth_id': auth,
                    'text_id': text_loc
                }
                i += 1
        self.author_list = list(self.data.keys())

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        auth_id = self.idx_to_txt_map[item]['auth_id']
        txt_num = self.idx_to_txt_map[item]['text_id']
        text0 = self.data[auth_id][txt_num]
        # just return the txt and the auth_id????
        return InputExample(texts=[text0], label=auth_id)


def get_random_substring(txt, substr_len=512*5):
    if len(txt) > substr_len + 1:
        idx = random.randint(0, len(txt) - substr_len + 1)
        txt = txt[idx:idx+substr_len]
    return txt


class AVDataset(Dataset):

    def __init__(self, data_path):
        # , char_vocab=None, tok_vocab=None, char_to_id=None, tok_to_id=None, **kwargs):
        super(AVDataset, self).__init__()  # char_vocab, tok_vocab, char_to_id, tok_to_id, **kwargs)

        _data = get_av_dataset(data_path)
        self.data = []
        self.raw_data = []
        for label, text0, text1 in _data:
            self.data.append(InputExample(label=label, texts=[text0, text1]))
            self.raw_data.append([label, text0, text1])
        self.data_len = len(self.data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        return self.data[item]

    def get_chunked(self, item):
        # chunk each evaluation text so we can evaluate many pairs per document pair
        chunk_len = 256 * 5
        if isinstance(item, slice):

            samples = self.raw_data[item]
            chunked_samples = []
            for label, text0, text1 in samples:
                chunked_samples.append(self.break_sample_into_chunks(label, text0, text1, chunk_len))
            return chunked_samples
        else:
            label, text0, text1 = self.raw_data[item]
            return self.break_sample_into_chunks(label, text0, text1, chunk_len)

    @staticmethod
    def break_sample_into_chunks(lbl, txt0, txt1, chunk_len, max_txt_len=100000):
        txt0 = txt0[:max_txt_len]
        txt1 = txt1[:max_txt_len]
        chunked = []
        min_len = min(len(txt0), len(txt1))
        for i in range(0, min_len, chunk_len):
            chunked.append(InputExample(label=lbl, texts=[txt0[i:i + chunk_len], txt1[i:i + chunk_len]]))
        return chunked


def main():
    # get command line args
    parser = argparse.ArgumentParser(description='Get args for SiameseBert Authorship Attribution Evaluation')

    parser.add_argument('--wandb_project', type=str, default='test')
    parser.add_argument('--model_path', type=str, default='bert-base-cased')
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--output_path', type=str, default='SiameseBert_output')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--scheduler', type=str, default='warmuplinear',
                        help='The learning rate scheduler, available are: constantlr, warmupconstant, warmuplinear,'
                             'warmupcosine, warmupcosinenewwithhardrestarts')
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--which_loss_fn', type=str, default='ModifiedContrastiveLoss')
    parser.add_argument('--evaluation_steps', type=int, default=0)
    parser.add_argument('--distance_metric', type=str, default='euclidean')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--diff_margin', type=float, default=0.09)
    parser.add_argument('--same_margin', type=float, default=0.91)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--correct_bias', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--kernel_fn', action='store_true')
    parser.add_argument('--AA', action='store_true')
    parser.add_argument('--num_docs_per_auth_embedding', type=int, default=5)
    parser.add_argument('--num_chunks_per_doc_per_auth_embedding', type=int, default=10)
    parser.add_argument('--evaluate_first', action='store_true')
    parser.add_argument('--evaluate_only', action='store_true')
    parser.add_argument('--true_hard_negatives', action='store_true')
    parser.add_argument('--triplet_margin', type=float, default=5)
    parser.add_argument('--fit_threshold', action='store_true')
    args = parser.parse_args()

    # set the seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.model = 'SiameseBERT'

    with wandb.init(project=args.wandb_project, config=vars(args)):

        run_name = wandb.run.name

        if args.which_loss_fn in ['AllTriplet', 'BatchHardTriplet']:
            train_dataset = AAdataset_triplet(args.train_path)
        else:
            train_dataset = AADataset(args.train_path, true_hard_negatives=args.true_hard_negatives)

        if args.AA:
            test_dataset = get_aa_dataset(args.test_path)
            train_dset_for_authembs = get_aa_dataset(args.train_path)
        else:
            test_dataset = AVDataset(args.test_path)

        model_args = {'hidden_dropout_prob': args.dropout,
                      'attention_probs_dropout_prob': args.dropout}
        if args.model_path == 'bert-base-cased':
            word_embedding_model = models.Transformer(args.model_path,
                                                      model_args=model_args,
                                                      max_seq_length=args.max_seq_len)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_cls_token=True,
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_max_tokens=True)
            dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                       out_features=256,
                                       activation_function=nn.ReLU())
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model],
                                        device=args.device)  # , Dropout(args.dropout)],
        else:
            model = SentenceTransformer(model_name_or_path=args.model_path,
                                        device=args.device)  # , Dropout(args.dropout)],

        logging.debug('model info:')
        logging.debug(model)

        logging.info('building datasets')

        # which loss fn are are using?
        if args.distance_metric == 'euclidean':
            contrastive_dist_metric = SiameseDistanceMetric.EUCLIDEAN
            triplet_dist_metric = BatchHardTripletLossDistanceFunction.eucledian_distance

        elif args.distance_metric == 'cosine':
            contrastive_dist_metric = SiameseDistanceMetric.COSINE_DISTANCE
            triplet_dist_metric = BatchHardTripletLossDistanceFunction.cosine_distance

        else:
            assert False, f'{args.distance_metric} is not defined'

        if args.true_hard_negatives:
            train_dataset.reset_hard_negatives(model, triplet_dist_metric)

        # get the dataloaders
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

        if args.kernel_fn:
            kernel_fn = lambda x: torch.pow(np.e, -0.09 * torch.pow(x, 3)) + 1e-10
        # elif args.distance_metric == 'euclidean':
        #     kernel_fn = lambda x: -x
        else:
            kernel_fn = lambda x: -x

        post_eval_callable = None
        if args.true_hard_negatives:
            post_eval_callable = train_dataset.reset_hard_negatives

        if args.AA:
            evaluator = AaEvaluator(train_dataset=train_dset_for_authembs,
                                    test_dataset=test_dataset,
                                    distance_metric=contrastive_dist_metric,
                                    triplet_dist_metric=triplet_dist_metric,
                                    num_docs_per_auth_embedding=args.num_docs_per_auth_embedding,
                                    num_chunks_per_doc_per_auth_embedding=args.num_chunks_per_doc_per_auth_embedding,
                                    post_eval_callable=post_eval_callable
                                    )
        else:
            evaluator = ContrastiveEvaluator(test_dataset,
                                             batch_size=args.eval_batch_size,
                                             distance_metric=contrastive_dist_metric,
                                             triplet_dist_metric=triplet_dist_metric,
                                             kernel_fn=kernel_fn,
                                             post_eval_callable=post_eval_callable)

        if args.which_loss_fn == 'ContrastiveLoss':
            train_loss = MyContrastiveLoss(model=model, distance_metric=contrastive_dist_metric,
                                           margin=args.margin, kernel_fn=kernel_fn)

        elif args.which_loss_fn == 'ModifiedContrastiveLoss':
            train_loss = ModifiedContrastiveLoss(model=model, distance_metric=contrastive_dist_metric,
                                                 different_margin=args.diff_margin,
                                                 same_margin=args.same_margin,
                                                 kernel_fn=kernel_fn)
        elif args.which_loss_fn == 'AllTriplet':
            # this causes a bit of weirdness with the kernel function. Maybe we just do 1-kernelfn(x),
            #   then say margin is 0.2 or something?  But this kinda changes the metric. We can just
            #   leave as is, and set the margin such that we think it'll work with the kernel fn - yeah do that.
            train_loss = BatchAllTripletLoss(model=model, distance_metric=triplet_dist_metric, margin=args.triplet_margin)

        elif args.which_loss_fn == 'BatchHardTriplet':
            # same note on margin as AllTriplet
            train_loss = BatchHardTripletLoss(model=model, distance_metric=triplet_dist_metric, margin=args.triplet_margin)
        else:
            assert False, 'The selected loss function is not recognized.'

        # do we want to run an evaluation before training?
        if args.evaluate_first:
            logging.info('launching initial evaluation')
            model.eval()
            init_perf = evaluator(model=model,
                                  epoch=0,
                                  steps=0,
                                  batch_size=args.eval_batch_size)
            metric_name = 'macro_accuracy' if args.AA else 'AUC'
            logging.info(f'initial evaluation finished: {metric_name}={init_perf}')
            model.train()
            if args.evaluate_only:
                logging.info('evaluate_only was set to True, exiting.')
                exit(0)

        logging.info('training')

        # train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=args.epochs,
                  scheduler=args.scheduler,
                  warmup_steps=args.warmup_steps,
                  save_best_model=True,
                  output_path=os.path.join(args.output_path, run_name),
                  evaluator=evaluator,
                  evaluation_steps=args.evaluation_steps,
                  optimizer_params={'lr': args.lr, 'eps': args.eps, 'correct_bias': args.correct_bias},)
        if fitThreshold and not args.AA:
            threshold_finder = ContrastiveEvaluator(train_dataset,
                                             batch_size=args.eval_batch_size,
                                             distance_metric=contrastive_dist_metric,
                                             triplet_dist_metric=triplet_dist_metric,
                                             kernel_fn=kernel_fn,
                                             post_eval_callable=post_eval_callable)
            performance, threshold = threshold_finder(model=model,
                                               epoch=0,  
                                               steps=0,
                                               batch_size=args.eval_batch_size,
                                               fit_threshold=True)
            performance = evaluator(model=model,
                                  epoch=0,
                                  steps=0,
                                  batch_size=args.eval_batch_size,
                                  threshold=threshold)

        logging.info('finished')


if __name__ == "__main__":
    main()
