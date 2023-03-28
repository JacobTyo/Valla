from sentence_transformers.evaluation import SentenceEvaluator
import torch
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
import math
import time
import random
from tqdm import tqdm
from sentence_transformers.losses.ContrastiveLoss import SiameseDistanceMetric
import numpy as np
from typing import Optional
import logging
import wandb
from scipy import stats
from valla.utils.eval_metrics import av_metrics, aa_metrics

logging.basicConfig(level=logging.DEBUG)


class AaEvaluator(SentenceEvaluator):

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 batch_size=1,
                 distance_metric=BatchHardTripletLossDistanceFunction.eucledian_distance,
                 triplet_dist_metric=None,
                 median: bool = False,
                 num_docs_per_auth_embedding: int = 1,
                 num_chunks_per_doc_per_auth_embedding: int = 10,
                 post_eval_callable: callable = None
                 ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.distance_metric = distance_metric
        self.median = median
        self.num_docs_per_auth_embedding = num_docs_per_auth_embedding
        self.num_chunks_per_doc_per_auth_embedding = num_chunks_per_doc_per_auth_embedding

        self.avg_k_author_embeddings = {}  # this is key'd on author number with each entry being the averaged k chunk
        self.rdm_k_author_embeddings = {}  # this is key'd on author number with each entry being the k chunks from 1 doc
        self.knn_author_embeddings = {}  # this is key'd on author number with each entry being a chunk from N and k
        self.chunks = {}
        self.triplet_dist_metric = triplet_dist_metric
        self.post_eval_callable = post_eval_callable

    def get_chunks(self):
        for auth, text in self.train_dataset:
            if auth in self.chunks:
                if len(self.chunks[auth]) > self.num_docs_per_auth_embedding:
                    continue
            self.chunks.setdefault(auth, []).append(self.listify_text(text))

    def listify_text(self, txt, chunk_len=512, max_txt_len=100000):
        txt = txt[:max_txt_len]
        chunked = []
        for i in range(0, len(txt), chunk_len):
            if i > self.num_chunks_per_doc_per_auth_embedding:
                break
            chunked.append(txt[i:i + chunk_len])
        return chunked

    def get_author_embeddings(self, model):
        # we need to compare several things:
        #   1) pick a single text and use chunks of it as the k author embedding
        #   2) pic N texts and then average the first k chunks of each together to get k author embeddings
        #   3) pick N texts and k chunks, embed and keep all separate, then just do kNN to classify test text

        with torch.no_grad():

            # this builds self.chunks which contains the data we want to build the embeddings
            self.get_chunks()

            # so now embed all of the data in chunks to get the average list of author embeddings as a
            #   function of chunk number
            for auth, chunks in tqdm(self.chunks.items(), desc='building auth embeddings'):
                embeddings = []
                num_docs = 0
                for doc_num, doc_chunks in enumerate(chunks):
                    num_docs += len(doc_chunks)
                    embeddings.append(model.encode(doc_chunks, show_progress_bar=False))
                    if doc_num == 0:
                        self.rdm_k_author_embeddings[auth] = np.asarray(embeddings)
                embeddings = np.asarray(embeddings)
                # now ge the median doc encoding per chunk location
                self.avg_k_author_embeddings[auth] = np.mean(embeddings, axis=1)
                self.knn_author_embeddings[auth] = np.reshape(embeddings, newshape=(num_docs, -1))

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, batch_size: int = -1,
                 num_samples_per_auth_embedding: int = 1, median: bool = None) -> float:

        self.get_author_embeddings(model)

        avg_predictions = []
        rdm_predictions = []
        knn1_predictions = []
        knn3_predictions = []
        knn5_predictions = []
        labels = []

        for auth, text in tqdm(self.test_dataset, desc='AA evaluation'):

            labels.append(auth)

            # first, get the first k embeddings of the chunked text
            chunked_text = self.listify_text(text)
            text_embeddings = model.encode(chunked_text, show_progress_bar=False)

            # now compare the distances and make a prediction
            id_to_dist = {}
            rdm_id_to_dist = {}
            knn_id_to_dist = []
            for auth_num, author_embedding in self.avg_k_author_embeddings.items():
                rdm_auth_emb = self.rdm_k_author_embeddings[auth_num]
                knn_auth_emb = self.knn_author_embeddings[auth_num]
                # this is num_chunksxemb_size
                avg_distances = self.distance_metric(torch.tensor(author_embedding), torch.tensor(text_embeddings)).numpy()
                # this is 1xemb_size
                rdm_distances = self.distance_metric(torch.tensor(rdm_auth_emb), torch.tensor(text_embeddings)).numpy()
                knn_distances = self.distance_metric(torch.tensor(knn_auth_emb), torch.tensor(text_embeddings)).numpy()
                # now just predict the auth as the closeset
                id_to_dist[auth_num] = np.mean(avg_distances)
                rdm_id_to_dist[auth_num] = np.mean(rdm_distances)
                five_smallest_knn_dists = sorted(knn_distances)
                for d in five_smallest_knn_dists:
                    knn_id_to_dist.append([auth_num, d])

            # now pic the smallest average distance
            preds = [[d, i] for i, d in id_to_dist.items()]
            preds = sorted(preds, key=lambda x: x[0])
            avg_predictions.append(preds[0][1])

            rdm_preds = [[d, i] for i, d in rdm_id_to_dist.items()]
            rdm_preds = sorted(rdm_preds, key=lambda x: x[0])
            rdm_predictions.append(rdm_preds[0][1])

            knn1_preds = [[d, i] for i, d in knn_id_to_dist]
            knn1_preds = sorted(knn1_preds, key=lambda x: x[0])
            knn1_predictions.append(knn1_preds[0][1])

            knn3 = [i for d, i in knn1_preds[:3]]
            knn3_predictions.append(stats.mode(knn3).mode[0])

            knn5 = [i for d, i in knn1_preds[:5]]
            knn5_predictions.append(stats.mode(knn5).mode[0])

        results = aa_metrics(labels=labels, predictions=avg_predictions, raw_outputs=None, no_auc=True, prefix='avg/')
        results_rdm = aa_metrics(labels=labels, predictions=rdm_predictions, raw_outputs=None, no_auc=True, prefix='random/')
        results.update(results_rdm)
        results_knn1 = aa_metrics(labels=labels, predictions=knn1_predictions, raw_outputs=None, no_auc=True, prefix='knn1/')
        results.update(results_knn1)
        results_knn3 = aa_metrics(labels=labels, predictions=knn3_predictions, raw_outputs=None, no_auc=True, prefix='knn3/')
        results.update(results_knn3)
        results_knn5 = aa_metrics(labels=labels, predictions=knn5_predictions, raw_outputs=None, no_auc=True, prefix='knn5/')
        results.update(results_knn5)

        wandb.log(results)
        if self.post_eval_callable is not None:
            self.post_eval_callable(model, self.triplet_dist_metric)
        return results['avg/macro_accuracy']


class ContrastiveEvaluator(SentenceEvaluator):

    def __init__(self, eval_dataset, batch_size: int = 100,
                 distance_metric=SiameseDistanceMetric.EUCLIDEAN,
                 triplet_dist_metric=None,
                 name: str = '', kernel_fn: Optional = None,
                 post_eval_callable: callable = None):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.distance_metric = distance_metric  # lambda x, y: 1-F.cosine_similarity(x, y)
        self.batch_size = batch_size
        self.name = name
        self.kernel_fn = kernel_fn
        self.post_eval_callable = post_eval_callable
        self.triplet_dist_metric = triplet_dist_metric
        assert kernel_fn is not None, 'you must provide a kernel function to transform a distance into a similarity'

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, batch_size: int = -1, threshold: float = 0.5, fitThreshold: bool = False):
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.
        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """

        # just encode all of the samples, then use some comparison to determine if they are the same or different?
        # Can I just wrap this with the contrastive loss thing somehow? I think so, try it

        model.eval()

        if batch_size == -1:
            batch_size = self.batch_size
        print('evaluating')
        start_time = time.time()
        truth = []
        avg_distances = []
        median_distances = []
        first_distances = []
        random_distances = []

        with torch.no_grad():
            for i in tqdm(range(len(self.eval_dataset))):
                # this returns a list of input examples, where each input example is from the same evaluation pair
                chunked = self.eval_dataset.get_chunked(i)

                chunked_emb1 = []
                chunked_emb2 = []
                chunked_truth = []  # this will all be the same value
                if len(chunked) > batch_size:
                    # batch the chunks
                    for batch_chunk_idx in range(0, len(chunked), batch_size):
                        additional = batch_size if batch_chunk_idx + batch_size < len(chunked) else len(chunked)-batch_chunk_idx
                        batch1 = [sample.texts[0] for sample in chunked[batch_chunk_idx:batch_chunk_idx+additional]]
                        batch2 = [sample.texts[1] for sample in chunked[batch_chunk_idx:batch_chunk_idx+additional]]
                        labels = [sample.label for sample in chunked[batch_chunk_idx:batch_chunk_idx+additional]]
                        chunked_emb1.extend(model.encode(batch1, show_progress_bar=False))
                        chunked_emb2.extend(model.encode(batch2, show_progress_bar=False))
                        chunked_truth.extend(labels)
                else:
                    batch1 = [sample.texts[0] for sample in chunked]
                    batch2 = [sample.texts[1] for sample in chunked]
                    labels = [sample.label for sample in chunked]
                    chunked_emb1.extend(model.encode(batch1, show_progress_bar=False))
                    chunked_emb2.extend(model.encode(batch2, show_progress_bar=False))
                    chunked_truth.extend(labels)

                distances = self.distance_metric(torch.tensor(chunked_emb1), torch.tensor(chunked_emb2)).numpy()
                # so this is a list of predictions that compares all the parts of the texts together
                # so what we need to do is to predict the median prediction, but for auc what do we do?
                # we are going to do 3 things:
                #  1) just get the first one and call it our guess
                #  2) average all of the distances together
                #  3) take the median distance
                #  4) take a random location as our distance

                avg_distances.append(np.mean(distances))
                median_distances.append(np.median(distances))
                first_distances.append(distances[0])
                random_distances.append(random.choice(distances))
                truth.append(chunked_truth[0])

        avg_similarities = self.kernel_fn(torch.tensor(avg_distances, requires_grad=False)).numpy()
        med_similarities = self.kernel_fn(torch.tensor(median_distances, requires_grad=False)).numpy()
        fst_similarities = self.kernel_fn(torch.tensor(first_distances, requires_grad=False)).numpy()
        rdm_similarities = self.kernel_fn(torch.tensor(random_distances, requires_grad=False)).numpy()

        overall_best_auc = 0
        overall_best_method = ''
        results = {}
        sims = {}
        
        for sim, sim_name in [[avg_similarities, 'mean/'],
                              [med_similarities, 'median/'],
                              [fst_similarities, 'first/'],
                              [rdm_similarities, 'random/']]:
            sim_stats = {
                f'{sim_name}similarity_min': np.min(sim),
                f'{sim_name}similarity_max': np.max(sim),
                f'{sim_name}similarity_mean': np.mean(sim),
                f'{sim_name}similarity_std': np.std(sim),
                f'{sim_name}similarity_median': np.median(sim),
            }

            #results.update(av_metrics(labels=truth, probas=sim, threshold=0.5, prefix=sim_name))
            if fitThreshold:
                threshold = sim.mean()
                sims[f'{sim_name}threshold'] = np.median(sim)
            else:
                if isinstance(threshold, float):
                    results.update(av_metrics(labels=truth, probas=sim, threshold=threshold, prefix=sim_name))
                else:
                    results.update(av_metrics(labels=truth, probas=sim, threshold=threshold[f'{sim_name}threshold'], prefix=sim_name))
                sims.update(sim_stats)
        if not fitThreshold:
            wandb.log(results)
            wandb.log(sims)

        model.train()

        eval_time_delta = time.time() - start_time
        logging.info(f'Evaluation took {eval_time_delta}')
        logging.info(f'the best performing evaluation method was {overall_best_method}')

        if self.post_eval_callable is not None:
            self.post_eval_callable(model, self.triplet_dist_metric)
        if fitThreshold:
            return overall_best_auc, sims
        return overall_best_auc

