# Dataset Access: https://huggingface.co/datasets/blog_authorship_corpus/blob/main/blog_authorship_corpus.py
from valla.utils.dataset_utils import finalize_dataset, list_dset_to_dict, auth_text_make_unique
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
import os
import argparse
import json
import numpy as np
import random

logging.basicConfig(level=logging.INFO)


def get_posts_from_file(pth):
    # variable to store the posts of this specific author
    auth_content = []

    with open(pth, 'r', errors='ignore') as f:
        # we want just the data between the <post> and </post> tags as content
        # get all lines - this contains all posts of a single author,
        #   so there are many posts between the <post> tags
        lines = f.readlines()

        # a variable to track if we are between the <post> tags or not
        get_content = False

        # variable to store the information of a single post
        this_line_content = ''

        # step through the file
        for line in lines:

            # check date
            if '<date>' in line:
                date = line.strip().replace('<date>', '').replace('</date>', '').strip()

            if get_content:

                # this means the blog post has concluded, so add the this_line_content string to the
                #   auth_content variable as a new blog post (track the date as well just in case we need later)
                if '</post>' in line:
                    get_content = False
                    this_line_content += line.replace('</post>', '').strip()
                    auth_content.append((this_line_content.strip(), date))
                    this_line_content = ''
                elif not line.isspace():
                    this_line_content += (line.strip() + ' ').replace('\0', '')

            else:

                # only get what is between these lines. XML parsing may be easier but this will work
                if '<post>' in line:
                    get_content = True
                    this_line_content += line.replace('<post>', '').strip()
    return auth_content


def process_blogs_xmls(dataset_path: str) -> (dict, list):
    # get the dataset as a dic, with key being author id and value being a list of contents
    raw_data, non_filtered_data = {}, {}
    author_works_counts = []
    nf_works_counts = []

    # lots of duplicated texts - count them
    dup_texts, text_count, unique_text_count = 0, 0, 0
    unique_texts = {}

    # transform user_id's as well
    label_transformer, label_to_id_transformer = {}, {}
    label_count = 0

    # for every blog (i.e. every file in the directory)
    for directory, subdirectories, files in tqdm(os.walk(dataset_path)):
        for file in tqdm(files):
            # get the author information
            meta_info = file.split('.')
            auth_id = meta_info[0]
            gender = meta_info[1]
            age = meta_info[2]
            topic = meta_info[3]
            sign = meta_info[4]

            auth_content = get_posts_from_file(os.path.join(directory, file))

            # we have all the blog posts of the current author, add all of this information to the dictionary
            this_auth_text_num = 0
            for auth_c in auth_content:
                # This is slow, but we need to make sure we don't add duplicated texts (71,298 of them initially)
                t = auth_c[0].strip()
                if len(t) < 100:
                    continue
                t = t[:100000]

                # transform author_id into integers
                if auth_id not in label_transformer.keys():
                    label_transformer[auth_id] = label_count
                    label_to_id_transformer[label_count] = auth_id
                    label_count += 1

                text_count += 1
                if t in unique_texts:
                    # skip this text but just track for now
                    unique_texts[t].append(label_transformer[auth_id])
                else:
                    unique_texts[t] = [label_transformer[auth_id]]
                    unique_text_count += 1
                    this_auth_text_num += 1
                    # only add if long enough

                    if label_transformer[auth_id] in raw_data.keys():
                        # raw_data[label_transformer[auth_id]].append
                        #   ((auth_c[0], auth_id, gender, age, topic, sign, auth_c[1]))
                        raw_data[label_transformer[auth_id]].add(t)
                    else:
                        raw_data[label_transformer[auth_id]] = {t}

                non_filtered_data.setdefault(label_transformer[auth_id], []).append(t)

            # track the number of works each author has
            try:
                author_works_counts.append([label_transformer[auth_id], this_auth_text_num])
                nf_works_counts.append([label_transformer[auth_id], len(non_filtered_data[label_transformer[auth_id]])])
            except Exception as e:
                logging.warning(f'The following exception was caught in handling the author counts: {e}')

        print(f'total texts: {text_count}')
        print(f'duplicate texts: {text_count - unique_text_count}')

        for auth in raw_data.keys():
            raw_data[auth] = set(raw_data[auth])

        return raw_data, author_works_counts, non_filtered_data, nf_works_counts, label_to_id_transformer, unique_texts


if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of the blogs dataset')

    parser.add_argument('--dataset_path', type=str, default='/home/jtyo/data/Projects/'
                                                            'On_the_SOTA_of_Authorship_Verification/datasets/'
                                                            'Blog/raw/blogs')
    parser.add_argument('--dataset_save_path', type=str, default='/home/jtyo/data/Projects/'
                                                                 'On_the_SOTA_of_Authorship_Verification/'
                                                                 'datasets/Blog/processed')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # get the blogs dataset as a dictionary
    logging.info(f'getting the original dataset file from {args.dataset_path}')
    data, author_counts, non_deduped_data, nd_counts, label_dict, shared_text_to_id = process_blogs_xmls(args.dataset_path)
    # data = auth_text_make_unique(data)

    # now we will build two datasets, blogs50, and blogsAV.
    #   The blogs 50 dataset will be comprised of the top 50 authors (in terms of number of posts),
    logging.info(f'getting the top 50 authors based on the number of blog posts')
    top50_authors = [auth_num for auth_num, _ in sorted(author_counts, key=lambda x: x[1], reverse=True)][:50]
    top50_nondedupted = [auth_num for auth_num, _ in sorted(nd_counts, key=lambda x: x[1], reverse=True)][:50]

    duped_id_list = {}

    # just see how many texts are duplicated in the top50_nondedupted
    non_deduped_check = {}
    for person in top50_nondedupted:
        for text in non_deduped_data[person]:
            if text not in non_deduped_check.keys():
                non_deduped_check[text] = 1
            else:
                non_deduped_check[text] += 1

    # now print the number of texts duplicated
    duplicated_texts_og, dt = 0, []
    for t, cnt in non_deduped_check.items():
        if cnt > 1:
            duplicated_texts_og += 1
            dt.append(t)

    logging.info(f'there were {duplicated_texts_og} duplicated texts in the original top 50')
    logging.info(f'here are a couple')
    logging.info(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    logging.info(f'{dt[0]}')
    logging.info(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    logging.info(f'{dt[23]}')
    logging.info(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    logging.info(f'{dt[1039]}')
    logging.info(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # saving the duplicated texts to a file just incase
    with open('../docs/blogs_dup_texts.txt', 'w') as f:
        for t in dt:
            f.write(t)
            f.write('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')

    total_duped_auth_change = 0
    total_texts_nondedupted = 0
    total_texts_dedupted = 0

    for person in top50_authors:
        total_texts_dedupted += len(data[person])

    for person in top50_nondedupted:
        total_texts_nondedupted += len(non_deduped_data[person])
        if person not in top50_authors:
            total_duped_auth_change += 1
            try:
                tmp = len(non_deduped_data[person])
                tmp2 = len(data[person])
                shared_authors = []
                for v in shared_text_to_id.values():
                    if person in v:
                        for p in v:
                            if p != person and label_dict[p] not in shared_authors:
                                shared_authors.append(label_dict[p])

                s = f'\n\nAuthor {label_dict[person]} was in the non-deduplicated version but not in the deduplicated ' \
                    f'version! \n the other authors with this text are {shared_authors}' \
                    f'\nThis author has {tmp-tmp2} non-deduplicated texts. '
                duped_id_list[label_dict[person]] = shared_authors
                # v = f'texts non_deduped: {tmp}, deduped: {tmp2}'
                # t = f'{non_deduped_data[person][0]}' \
                #     f'{non_deduped_data[person][1]}' \
                #     f'{non_deduped_data[person][2]}' \
                #     f'{non_deduped_data[person][50]}'
                logging.warning(s)
                # logging.warning(t)
                with open('../docs/blogs_warnings.err', 'a') as f:
                    f.write(s)
                    # f.write(v)
                    # f.write(t)
            except KeyError:
                tmp = len(non_deduped_data[person])
                shared_authors = []
                for v in shared_text_to_id.values():
                    if person in v:
                        for p in v:
                            if p != person and label_dict[p] not in shared_authors:
                                shared_authors.append(label_dict[p])
                s = f'\n\nAuthor {label_dict[person]} was not in the "deduplicated" dataset, which means it was a ' \
                    f'duplicated author in the dataset, so it was removed. ' \
                    f'\n The author had {tmp} texts' \
                    f'\n The other authors with this text are {shared_authors}'
                duped_id_list[label_dict[person]] = shared_authors
                logging.warning(s)
                with open('../docs/blogs_warnings.err', 'a') as f:
                    f.write(s)

    logging.warning(f'there were {total_texts_nondedupted} total texts without duplication')
    logging.warning(f'there were {total_texts_dedupted} total texts after removing duplicates')
    logging.warning(f'{total_duped_auth_change} authors were removed after deduplication')

    # now determine how many texts changed from non deduped to deduped
    ts = []
    removed_texts = 0
    for p in top50_authors:
        ts.extend(list(data[p]))
    assert len(ts) == len(list(set(ts)))
    for p in top50_nondedupted:
        for t in non_deduped_data[p]:
            if t not in ts:
                removed_texts += 1

    non_deduped_texts = 0
    for p in top50_nondedupted:
        non_deduped_texts += len(non_deduped_data[p])
    logging.warning(f'there were {removed_texts} texts removed after deduplication')
    logging.warning(f'there are {len(ts)} texts in the deduplicated training set')
    logging.warning(f'there are {non_deduped_texts})')

    exit(0)

    with open('../docs/blogs_duplist.json', 'w') as f:
        json.dump(duped_id_list, f, sort_keys=True, indent=4)

    # auth_num normalizer
    auth_num_to_idx = {}
    # all samples
    all_data = []
    top50_dict = {}
    for i, auth in enumerate(top50_authors):
        logging.info(f'author: {auth} ({i}) has {len(data[auth])} texts')
        for text in data[auth]:
            all_data.append([i, text])
            top50_dict.setdefault(i, []).append(text)

    logging.info(f'splitting the data into train/eval/test sets')
    # now split into stratified train(60%)/val(20%)/test(20%) splits
    train_set, eval_and_test_set = train_test_split(all_data, test_size=0.4, shuffle=True, random_state=args.seed,
                                                    stratify=[lbl for lbl, _ in all_data])
    aa_eval, aa_test = train_test_split(eval_and_test_set, test_size=0.5, shuffle=True, random_state=args.seed,
                                          stratify=[lbl for lbl, _ in eval_and_test_set])

    finalize_dataset(top50_dict, list_dset_to_dict(train_set), list_dset_to_dict(aa_eval), list_dset_to_dict(aa_test),
                     dataset_name='blogs50', save_path=args.dataset_save_path)
