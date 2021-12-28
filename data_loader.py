import json
import random
from tqdm import tqdm
import logging

def train_generate_simple(dataset, batch_size, few, symbol2id):
    logging.info('LOADING TRAINING DATA')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    logging.info('LOADING CANDIDATES')
    rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
    task_pool = list(train_tasks.keys())
    num_tasks = len(task_pool)
    rel_idx = 0

    while True:
        if rel_idx % num_tasks == 0:
            random.shuffle(task_pool)
        query = task_pool[rel_idx % num_tasks]
        rel_idx += 1
        candidates = rel2candidates[query]
        train_and_test = train_tasks[query]
        random.shuffle(train_and_test)
        support_triples = train_and_test[:few]
        support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]  

        all_test_triples = train_and_test[few:]
        if len(all_test_triples) < batch_size:
            query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
        else:
            query_triples = random.sample(all_test_triples, batch_size)
        query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

        false_pairs = []
        for triple in query_triples:
            e_h = triple[0]
            e_t = triple[2]
            while True:
                noise = random.choice(candidates)
                if noise != e_t:
                    break
            false_pairs.append([symbol2id[e_h], symbol2id[noise]])

        yield support_pairs, query_pairs, false_pairs

def train_generate(dataset, batch_size, few, symbol2id, ent2id, e1rel_e2, num_neg=1):
    logging.info('LOADING TRAINING DATA')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    logging.info('LOADING CANDIDATES')
    rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
    task_pool = list(train_tasks.keys())
    num_tasks = len(task_pool)
    rel_idx = 0

    while True:
        if rel_idx % num_tasks == 0:
            random.shuffle(task_pool)
        query = task_pool[rel_idx % num_tasks]
        rel_idx += 1
        candidates = rel2candidates[query]
        train_and_test = train_tasks[query]
        random.shuffle(train_and_test)
        support_triples = train_and_test[:few]
        support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]
        support_left = [ent2id[triple[0]] for triple in support_triples]
        support_right = [ent2id[triple[2]] for triple in support_triples]

        all_test_triples = train_and_test[few:]
        if len(all_test_triples) == 0:
            continue

        if len(all_test_triples) < batch_size:
            query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
        else:
            query_triples = random.sample(all_test_triples, batch_size)

        query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]
        query_left = [ent2id[triple[0]] for triple in query_triples]
        query_right = [ent2id[triple[2]] for triple in query_triples]

        false_pairs = []
        false_left = []
        false_right = []
        for triple in query_triples:
            e_h = triple[0]
            rel = triple[1]
            e_t = triple[2]
            while True:
                noise = random.choice(candidates)
                if (noise not in e1rel_e2[e_h+rel]) and noise != e_t:
                    break
            false_pairs.append([symbol2id[e_h], symbol2id[noise]])
            false_left.append(ent2id[e_h])
            false_right.append(ent2id[noise])

        yield support_pairs, query_pairs, false_pairs, support_left, support_right, query_left, query_right, false_left, false_right

def train_generate_(dataset, batch_size, few, symbol2id, ent2id, e1rel_e2, num_neg=1):
    logging.info('LOADING TRAINING DATA')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    logging.info('LOADING CANDIDATES')
    rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
    task_pool = list(train_tasks.keys())
    num_tasks = len(task_pool)
    rel_idx = 0

    while True:
        if rel_idx % num_tasks == 0:
            random.shuffle(task_pool)
        query = task_pool[rel_idx % num_tasks]
        rel_idx += 1
        candidates = rel2candidates[query]
        train_and_test = train_tasks[query]

        random.shuffle(train_and_test)
        support_triples = train_and_test[:few]
        support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]
        support_left = [ent2id[triple[0]] for triple in support_triples]
        support_right = [ent2id[triple[2]] for triple in support_triples]

        all_test_triples = train_and_test[few:]
        if len(all_test_triples) < batch_size:
            query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
        else:
            query_triples = random.sample(all_test_triples, batch_size)

        query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]
        query_left = [ent2id[triple[0]] for triple in query_triples]
        query_right = [ent2id[triple[2]] for triple in query_triples]

        labels = [1] * len(query_triples)

        # sample negative samples for every true triples
        for triple in query_triples:
            e_h = triple[0]
            e_t = triple[2]

            if e_t in candidates: candidates.remove(e_t)

            if len(candidates) >=num_neg:
                noises = random.sample(candidates, num_neg)
            else:
                noises = candidates

            for noise in noises:
                query_pairs.append([symbol2id[e_h], symbol2id[noise]])
                query_left.append(ent2id[e_h])
                query_right.append(ent2id[noise])
                labels.append(0)

        yield support_pairs, query_pairs, support_left, support_right, query_left, query_right, labels

def train_generate_matcher(dataset, batch_size, symbol2id):
    logging.info('LOADING TRAINING DATA')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    # logging.info('LOADING CANDIDATES')
    # rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
    positive_company = []
    negative_company = []
    positive_company_dict = {}
    negative_company_dict = {}
    for project, company_details in train_tasks.items():
        for company, details in company_details.items():
            if details['投标_项目_公司_是否_中标'] == 'false':
                details['投标_公司'] = company
                details['投标_项目'] = project
                negative_company.append(details)
                if project not in negative_company_dict:
                    negative_company_dict[project] = [details]
                else:
                    negative_company_dict[project] += [details]
            else:
                details['投标_公司'] = company
                details['投标_项目'] = project
                positive_company.append(details)
                if project not in positive_company:
                    positive_company_dict[project] = [details]
                else:
                    positive_company_dict[project] += [details]

    # Task 屬於每一個招標項目以及投標公司相關特徵
    task_pool = list(positive_company_dict.keys())
    num_tasks = len(task_pool)
    rel_idx = 0
    support_pairs, false_pairs, query_pairs = [], [], []
    while True:
        # 每次走完全部task shuffle training data 一遍
        while len(support_pairs) < batch_size:#for i in range(batch_size):
            if rel_idx % num_tasks == 0:
                random.shuffle(task_pool)
            # Support 應該是機構以及其相關特徵
            # Query 是項目
            # False 是未中標項目
            query = task_pool[rel_idx % num_tasks]
            rel_idx += 1
            if query in negative_company_dict:
                query_pairs.append(symbol2id[query])
                support_pair = [[symbol2id[relation], symbol2id[tail_entity]] for relation, tail_entity in positive_company_dict[query][0].items() if relation != '投标_项目_公司_是否_中标' and relation != '投标_项目_公司_本次_评标_排名']
                support_pairs.append(support_pair)
                false_pair = [[symbol2id[relation], symbol2id[tail_entity]] for relation, tail_entity in random.choice(negative_company_dict[query]).items() if relation != '投标_项目_公司_是否_中标' and relation != '投标_项目_公司_本次_评标_排名']
                false_pairs.append(false_pair)
            else:
                continue
        yield support_pairs, query_pairs, false_pairs




