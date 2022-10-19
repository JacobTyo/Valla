"""
adapted from https://github.com/GBarlas/mhc
"""
import argparse
import torch
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from simpletransformers.language_representation import RepresentationModel
from valla.dsets.loaders import aa_as_pandas, get_aa_dataset
from valla.utils.eval_metrics import aa_metrics
import logging
import wandb

logging.basicConfig(level=logging.INFO)


class MHC(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear_io = torch.nn.Linear(
            in_features=input_size,
            out_features=output_size,
            bias=True)

    def forward(self, input):
        return self.linear_io(input)


def run_mhc_sweep(config=None, train_pth=None, val_path=None, test_pth=None, device=None, project=None,
                  cache_dir=None, save_path=None):
    # start the wandb sweep
    # add some stuff to config, like paths and whatnot
    with wandb.init(config=config):
        config = wandb.config
        print(type(config))

        config.train_dataset = train_pth
        config.eval_dataset = val_path
        config.test_dataset = test_pth
        config.device = device
        config.model = 'AA_MHC'
        config.cache_dir = os.path.join(cache_dir, project)
        config.experiment_name = project
        config.no_save = True
        run_name = wandb.run.name
        config.output_filename = '_'.join([project, run_name, 'mhc_results.txt'])
        config.save_model_path = os.path.join(save_path, project, run_name)
        config.use_cuda = True
        config.max_seq_len = 512
        config.seed = 0
        wandb.config.update(config)

        main_mhc(config)


def run_mhc(params):
    wandb.login()

    wandb_project = params.experiment_name

    params.model = 'AA_MHC'

    params.output_filename = '_'.join([params.experiment_name, params.output_filename])

    params.cache_dir = os.path.join(params.cache_dir, params.experiment_name)
    params.save_model_path = os.path.join(params.save_model_path, params.experiment_name)

    wandb.init(project=wandb_project, config=vars(params))

    main_mhc(params)


def main_mhc(params):
    logging.info('getting dataset - checking for cached version')
    # check for cached version
    cache_paths = {
        'train': os.path.join(params.cache_dir, f'train_{params.experiment_name}.pkl'),
        'eval': os.path.join(params.cache_dir, f'eval_{params.experiment_name}.pkl'),
        'test': os.path.join(params.cache_dir, f'test_{params.experiment_name}.pkl')
    }
    if os.path.isfile(cache_paths['train']):
        # load the cached version of everything
        logging.info('loading the data from caches')
        df_train = pd.read_pickle(cache_paths['train'])
        df_eval = pd.read_pickle(cache_paths['eval'])
        df_test = pd.read_pickle(cache_paths['test'])

    else:
        logging.info('building dataset from scratch')
        df_train = aa_as_pandas(get_aa_dataset(params.train_dataset))
        df_eval = aa_as_pandas(get_aa_dataset(params.eval_dataset))
        df_test = aa_as_pandas(get_aa_dataset(params.test_dataset))

        train_len = len(df_train)
        eval_len = len(df_eval)
        test_len = len(df_test)

        logging.info('getting bert')
        # this model truncates the input samples at max_seq_len
        bert = RepresentationModel(
            model_type="bert",
            model_name="bert-base-uncased",
            use_cuda=params.use_cuda,
            cuda_device=params.device,
            args={
                "max_seq_length": params.max_seq_len
            }
        )

        logging.info('building tokens')
        df_train['input_ids'] = bert.tokenizer(df_train['text'].tolist())['input_ids']
        df_eval['input_ids'] = bert.tokenizer(df_eval['text'].tolist())['input_ids']
        df_test['input_ids'] = bert.tokenizer(df_test['text'].tolist())['input_ids']

        logging.info('building embeddings')

        # weird issue so using this as a workaround (inefficient)

        def add_embeddings_to_df(df, set_name=None):
            embeddings, shapes = [], []
            for i in tqdm(range(len(df)), desc=f'embed {set_name}'):
                t = bert.encode_sentences([df['text'].iloc[i]], combine_strategy=None)
                embeddings.append(t[0])
                shapes.append(t.shape[1:])
            df['embedded'] = embeddings
            df['embed.shape'] = shapes
            return df

        df_train = add_embeddings_to_df(df_train, 'train')
        df_eval = add_embeddings_to_df(df_eval, 'eval')
        df_test = add_embeddings_to_df(df_test, 'test')

        assert train_len == len(df_train), 'the length of the train dataframe changed'
        assert eval_len == len(df_eval), 'the length of the eval dataframe changed'
        assert test_len == len(df_test), 'the length of the test dataframe changed'

        logging.info('caching the dataframes')
        # cache the dataframe
        if not os.path.exists(params.cache_dir):
            os.makedirs(params.cache_dir, exist_ok=True)
        df_train.to_pickle(cache_paths['train'])
        df_eval.to_pickle(cache_paths['eval'])
        df_test.to_pickle(cache_paths['test'])

    device = torch.device(params.device)
    # define seed
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)

    logging.info('getting tokens for the vocabulary')
    # get vocabulary from train set - select most frequent params.vocab_size tokens for use
    # token2id dict
    tokens_freq = {}
    for tokens in df_train['input_ids']:
        for token in tokens:
            tokens_freq[token] = tokens_freq.setdefault(token, 0) + 1

    tmp_dict = {}
    for i, (k, v) in enumerate(sorted(tokens_freq.items(), key=lambda t: t[1], reverse=True)):
        if i < params.vocab_size:
            tmp_dict[k] = v

    tokens = tmp_dict.keys()
    tokens_tensor = torch.LongTensor(list([i] for i in range(len(tokens)))).to(device)
    token2id = dict(zip(tokens, [i for i in range(len(tokens))]))

    vocab_size = params.vocab_size
    logging.info('building model')
    model = []
    optimizer = []
    n_authors = df_train['labels'].nunique()
    logging.debug(f'the number of authors in the train set is {n_authors}')
    input_size = 768  # df_train['embedded'].iloc[0].shape[-1]
    output_size = vocab_size
    for i_author in range(n_authors):
        model.append(MHC(input_size, output_size))
        model[-1].to(device)
        optimizer.append(torch.optim.Adagrad(model[-1].parameters()))

    results_file = Path(params.output_filename)
    if results_file.exists():
        print("File \"%s\" already exists! Results will be appended to the existing file." % results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    save_model_path = Path(params.save_model_path, results_file.stem)
    save_model_path.mkdir(parents=True, exist_ok=True)

    logging.info('launching training')
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    epoch = 0
    for epoch in range(epoch, params.epochs):
        print('Epoch: %d' % epoch)

        n_vectors = {}
        for set_type, data in zip(['train', 'eval', 'test'], [df_train, df_eval, df_test]):

            training = set_type == 'train'
            if training:
                for i_author in range(n_authors):
                    model[i_author].train()
            else:
                for i_author in range(n_authors):
                    model[i_author].eval()

            counters = []
            author_ids = []
            loss_per_author = torch.zeros(n_authors, len(data)).to(device)
            i2counter = {}
            for i, (author_id, tokens, representations) in enumerate(
                    tqdm(zip(data['labels'], data['input_ids'], data['embedded']), desc='%s set' % set_type,
                         total=len(data))):

                representations = torch.Tensor(representations).to(device)
                author_ids.append(author_id)
                temp_loss_per_author = torch.zeros(n_authors).to(device)
                n = 0
                # TODO: add batching here for speedup
                for token, feature in zip(tokens[1:], representations):
                    if token not in token2id:
                        continue
                    n += 1

                    if training:
                        output = model[author_id].forward(feature)
                        En = torch.nn.functional.cross_entropy(output.reshape(1, vocab_size),
                                                               tokens_tensor[token2id[token]], reduction='none')
                        temp_loss_per_author[author_id] += En.data[0]

                        En.backward()
                        optimizer[author_id].step()
                        model[author_id].zero_grad()
                        continue
                    else:
                        for i_author in range(n_authors):
                            output = model[i_author].forward(feature)
                            En = torch.nn.functional.cross_entropy(output.reshape(1, vocab_size),
                                                                   tokens_tensor[token2id[token]], reduction='none')
                            temp_loss_per_author[i_author] += En.data[0]

                loss_per_author[:, i] += (temp_loss_per_author / n)
                wandb.log({"loss_per_author": loss_per_author})

            if set_type == 'train':
                n_vectors['None'] = torch.zeros_like(loss_per_author.mean(1))
                logging.debug(f'finished with train set on epoch {epoch}')
                logging.debug(f"n_vectors['None'].shape = {n_vectors['None'].shape}")
                continue
            elif set_type == 'eval':
                n_vectors['C'] = loss_per_author.mean(1)
                logging.debug(f'finished with eval set on epoch {epoch}')
                logging.debug(f"n_vectors['C'].shape = {n_vectors['C'].shape}")
                continue
            elif set_type == 'test':
                n_vectors['U'] = loss_per_author.mean(1)
                logging.debug(f'finished with test set on epoch {epoch}')
                logging.debug(f"n_vectors['U'].shape = {n_vectors['U'].shape}")

            # save results
            logging.debug('writing results to output file and wandb')
            logging.debug(f'len(author_ids) = {len(author_ids)}')
            logging.debug(f'type(author_ids[0]) = {type(author_ids[0])}')
            with results_file.open('a') as fa:
                print("Epoch=%d" % epoch, file=fa)
                for C, n_vector in n_vectors.items():
                    lbls, predictions, raw_out = [], [], []
                    print("  NormalizationCorpus=%s" % C, file=fa),
                    print("  NormalizationVector=%s" % ','.join([str(float(x)) for x in n_vector]), file=fa)
                    for a, l in zip(author_ids, (loss_per_author.t() - n_vector)):
                        # print("    Counter=%s" % n, file=fa)
                        print("    Author=%d" % a, file=fa)
                        e_raw = l.cpu().detach().numpy()
                        e, r = l.sort(descending=False)
                        print("    Rank=%s" % ','.join([str(int(x)) for x in r]), file=fa)
                        predictions.append(r[0].cpu().detach().numpy())
                        lbls.append(a)
                        raw_out.append(e_raw)
                        print("    NormalizedError=%s" % ','.join([str(float(x)) for x in e]), file=fa)
                    print("  End=%s" % C, file=fa)
                    # log to wandb
                    results = aa_metrics(lbls, predictions, raw_out, prefix=f'{set_type}_{C}/', no_auc=True)
                    results['epoch'] = epoch
                    results['set_type'] = set_type
                    wandb.log(results)

        # save model
        if not params.no_save:
            for i in range(n_authors):
                this_mdl_path = os.path.join(save_model_path, f'mhc_{i}_{epoch}.model')
                state = {
                    'epoch': epoch,
                    'model': model[i].state_dict(),
                    'optimizer': optimizer[i].state_dict()
                }
                torch.save(state, this_mdl_path)
                wandb.save(this_mdl_path)


if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser(description='Run a N-Gram model from the command line')

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

    parser.add_argument('-o', '--output-filename', default='mhc.results')
    parser.add_argument('--epochs', type=int, default=21)
    parser.add_argument('--save-model-path', help="Path to save the model.", default='saved_models')
    parser.add_argument('--vocab_size', type=float, default=1000)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache_dir', type=str, default='mhc-cache')
    parser.add_argument('--no_save', action='store_true')

    args = parser.parse_args()

    run_mhc(args)
