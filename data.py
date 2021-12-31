'''
prepare the triples for few-shot training
'''
import numpy as np
from collections import defaultdict
import json

from time import time

# def build_vocab(dataset):
#     rels = set()
#     ents = set()
#
#     with open(dataset + '/path_graph') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.rstrip()
#             #e1 rel e2
#             rel = line.split('\t')[1]
#             e1 = line.split('\t')[0]
#             e2 = line.split('\t')[2]
#             rels.add(rel)
#             rels.add(rel + '_inv')
#             ents.add(e1)
#             ents.add(e2)
#
#     relationid = {}
#     for idx, item in enumerate(list(rels)):
#         relationid[item] = idx
#     entid = {}
#     for idx, item in enumerate(list(ents)):
#         entid[item] = idx
#
#     json.dump(relationid, open(dataset + '/relation2ids', 'w', encoding='utf-8'), ensure_ascii=False)
#     json.dump(entid, open(dataset + '/ent2ids', 'w', encoding='utf-8'), ensure_ascii=False)

# def combine_vocab(rel2id_path, ent2id_path, symbol2id_path):
#     symbol_id = {}
#
#     print('LOADING SYMBOL2ID')
#     rel2id = json.load(open(rel2id_path))
#     ent2id = json.load(open(ent2id_path))
#
#     # print set(rel2id.keys()) & set(ent2id.keys()) # '' and 'OOV'
#     # print('LOADING EMBEDDINGS')
#     # rel_embed = np.loadtxt(rel_emb, dtype=np.float32)
#     # ent_embed = np.loadtxt(ent_emb, dtype=np.float32)
#
#     # assert rel_embed.shape[0] == len(rel2id.keys())
#     # assert ent_embed.shape[0] == len(ent2id.keys())
#
#     i = 0
#     # embeddings = []
#     for key in rel2id.keys():
#         if key not in ['','OOV']:
#             symbol_id[key] = i
#             i += 1
#             # embeddings.append(list(rel_embed[rel2id[key],:]))
#
#     for key in ent2id.keys():
#         if key not in ['', 'OOV']:
#             symbol_id[key] = i
#             i += 1
#             # embeddings.append(list(ent_embed[ent2id[key],:]))
#
#     symbol_id['PAD'] = i
#     # embeddings.append(list(np.zeros((rel_embed.shape[1],))))
#     # embeddings = np.array(embeddings)
#
#     # assert embeddings.shape[0] == len(symbol_id.keys())
#     #np.savetxt(symbol2vec_path, embeddings)
#     json.dump(symbol_id, open(symbol2id_path, 'w'))

# def freq_rel_triples(dataset):
#     known_rels = defaultdict(list)
#     with open(dataset + '/path_graph') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.rstrip()
#             e1,rel,e2 = line.split()
#             known_rels[rel].append([e1,rel,e2])
#
#     train_tasks = json.load(open(dataset + '/train_tasks.json'))
#
#     for key, triples in train_tasks.items():
#         known_rels[key] = triples
#
#     json.dump(known_rels, open(dataset + '/known_rels.json', 'w'))
#
# def wiki_candidate(dataset):
#     ent2id = json.load(open(dataset + '/ent2ids'))
#     type2ents = defaultdict(list)
#     ent2type = {}
#     with open(dataset + '/instance_of') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.rstrip()
#             type_ = line.split()[2]
#             ent = line.split()[0]
#             if ent in ent2id:
#                 type2ents[type_].append(ent)
#                 ent2type[ent] = type_
#
#
#     train_tasks = json.load(open(dataset + '/train_tasks.json'))
#     dev_tasks = json.load(open(dataset + '/dev_tasks.json'))
#     test_tasks = json.load(open(dataset + '/test_tasks.json'))
#
#     all_reason_relations = train_tasks.keys() + dev_tasks.keys() + test_tasks.keys()
#
#     all_reason_relation_triples = train_tasks.values() + dev_tasks.values() + test_tasks.values()
#
#     print('How many few-shot relations', len(all_reason_relations))
#
#     rel2candidates = {}
#     for rel, triples in zip(all_reason_relations, all_reason_relation_triples):
#
#         possible_types = []
#         for example in triples:
#             possible_types.append(ent2type[example[2]])
#
#         possible_types = set(possible_types)
#
#         candidates = []
#         for _ in possible_types:
#             candidates += type2ents[_]
#
#         candidates = list(set(candidates))
#         if len(candidates) > 5000:
#             candidates = candidates[:5000]
#         rel2candidates[rel] = candidates
#
#     json.dump(rel2candidates, open(dataset + '/rel2candidates.json', 'w'))
#
#     dev_tasks_ = {}
#     test_tasks_ = {}
#
#     for key, triples in dev_tasks.items():
#         # print len(rel2candidates[key])
#         if len(rel2candidates[key]) < 20:
#             continue
#         dev_tasks_[key] = triples
#
#     for key, triples in test_tasks.items():
#         # print len(rel2candidates[key])
#         if len(rel2candidates[key]) < 20:
#             continue
#         test_tasks_[key] = triples
#
#     json.dump(dev_tasks_, open(dataset + '/dev_tasks.json', 'w'))
#     json.dump(test_tasks_, open(dataset + '/test_tasks.json', 'w'))

# def candidate_triples_backup(dataset):
#     '''
#     build candiate tail entities for every relation
#     '''
#     # calculate node degrees
#     # with open(dataset + '/path_graph') as f:
#     #     for line in f:
#     #         line = line.rstrip()
#     #         e1 = line.split('\t')[0]
#     #         e2 = line.split('\t')[2]
#
#     ent2ids = json.load(open(dataset+'/ent2ids'))
#
#     all_entities = ent2ids.keys()
#
#     type2ents = defaultdict(set)
#     for ent in all_entities:
#         try:
#             type_ = ent.split(':')[1]
#             type2ents[type_].add(ent)
#         except Exception as e:
#             continue
#
#     train_tasks = json.load(open(dataset + '/train_tasks.json'))
#     #train_tasks = json.load(open(dataset + '/known_rels.json'))
#     dev_tasks = json.load(open(dataset + '/dev_tasks.json'))
#     test_tasks = json.load(open(dataset + '/test_tasks.json'))
#
#     all_reason_relations = list(train_tasks.keys()) + list(dev_tasks.keys()) + list(test_tasks.keys())
#
#     all_reason_relation_triples = list(train_tasks.values()) + list(dev_tasks.values()) + list(test_tasks.values())
#
#     assert len(all_reason_relations) == len(all_reason_relation_triples)
#
#     rel2candidates = {}
#     for rel, triples in zip(all_reason_relations, all_reason_relation_triples):
#
#         possible_types = set()
#         for example in triples:
#             try:
#                 print(example[0].split(':'))
#                 print(example[1].split(':'))
#                 print(example[2].split(':'))
#                 exit()
#                 type_ = example[2].split(':')[1] # type of tail entity
#                 possible_types.add(type_)
#             except Exception as e:
#                 print(e)
#
#         candidates = []
#         for type_ in possible_types:
#             candidates += list(type2ents[type_])
#
#         rel2candidates[rel] = list(set(candidates))
#
#     json.dump(rel2candidates, open(dataset + '/rel2candidates.json', 'w'))

def build_vocab(dataset):
    relationid = {}
    with open(dataset + '/relation2id.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
        for item in list(data):
            if len(item.strip().split('\t')) != 2:
                continue
            name = item.strip().split('\t')[0]
            id = int(item.strip().split('\t')[1])
            relationid[name] = id

    entid = {}
    with open(dataset + '/entity2id.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
        for item in list(data):
            if len(item.strip().split('\t')) != 2:
                print(item)
                continue
            name = item.strip().split('\t')[0]
            id = int(item.strip().split('\t')[1])
            entid[name] = id

    json.dump(relationid, open(dataset + '/relation2ids', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(entid, open(dataset + '/ent2ids', 'w', encoding='utf-8'), ensure_ascii=False)

def for_filtering(dataset, save=False):
    e1rel_e2 = defaultdict(list)
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    dev_tasks = json.load(open(dataset + '/dev_tasks.json'))
    test_query_tasks = json.load(open(dataset + '/test_tasks_query.json'))
    test_support_tasks = json.load(open(dataset + '/test_tasks_support.json'))

    few_triples = []
    for _ in (list(train_tasks.values()) + list(test_query_tasks.values())):#+ list(dev_tasks.values()) + list(test_tasks.values())):
        few_triples += _

    for triple in few_triples:
        e1,rel,e2 = triple
        e1rel_e2[e1+rel].append(e2)

    clean_e1rel_e2 = defaultdict(list)
    for key in e1rel_e2:
        tmp = []

        for x in e1rel_e2[key]:
            if x not in tmp:
                tmp.append(x)

        clean_e1rel_e2[key] = tmp

    if save:
        json.dump(clean_e1rel_e2, open(dataset + '/e1rel_e2.json', 'w', encoding='utf-8'), ensure_ascii=False)

def candidate_triples(dataset):
    '''
    build candidate tail entities for every relation
    '''
    # calculate node degrees
    type2ents = defaultdict(set)
    with open(dataset + '/path_graph') as f:
        for line in f:
            line = line.rstrip()
            e1 = line.split('\t')[0]
            rel = line.split('\t')[1]
            e2 = line.split('\t')[2]
            type2ents[rel].add(e2)

    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    # dev_tasks = json.load(open(dataset + '/dev_tasks.json'))
    # test_query_tasks = json.load(open(dataset + '/test_tasks_query.json'))
    # test_support_tasks = json.load(open(dataset + '/test_tasks_support.json'))
    #
    all_reason_relations = list(train_tasks.keys()) #+ list(dev_tasks.keys()) #+ list(test_tasks.keys())
    # all_reason_relation_triples = list(train_tasks.values())+ list(test_query_tasks.values()) #+ list(dev_tasks.values()) #+ list(test_tasks.values())

    #assert len(all_reason_relations) == len(all_reason_relation_triples)

    rel2candidates = {}
    for rel in all_reason_relations:
        # possible_types = set()
        # for example in all_reason_relation_triples:
        #     # ['s400_雷达_系统', '使用_国家', '俄罗斯']
        #     try:
        #         print(example)
        #         exit()
        #         type_ = example[1] # type of tail entity
        #         possible_types.add(type_)
        #     except Exception as e:
        #         print(e)

        # candidates = []
        # for type_ in possible_types:
        candidates = list(type2ents[rel])
        rel2candidates[rel] = list(set(candidates))
        print(rel, len(list(set(candidates))))

    json.dump(rel2candidates, open(dataset + '/rel2candidates.json', 'w'), ensure_ascii=False)

def convert_vec(dataset):
    with open(dataset + '/transe.parameter.json') as f:
        data = json.load(f)
        for key in data.keys():
            if key == 'ent_embeddings.weight':
                ent_embeddings = data[key]
            elif key == 'rel_embeddings.weight':
                rel_embeddings = data[key]
    with open(dataset + '/entity2vec.TransE', 'w', encoding='utf-8') as f:
        for emb in ent_embeddings:
            for idx, e in enumerate(list(emb)):
                if idx == len(list(emb))-1:
                    f.write(str(e)+'\n')
                else:
                    f.write(str(e)+'\t')

    with open(dataset + '/relation2vec.TransE', 'w', encoding='utf-8') as f:
        for emb in rel_embeddings:
            for idx, e in enumerate(list(emb)):
                if idx == len(list(emb)) - 1:
                    f.write(str(e) + '\n')
                else:
                    f.write(str(e) + '\t')

def convert_quate_vec(dataset):
    with open(dataset + '/QuatE.json') as f:
        data = json.load(f)
        # dict_keys(
        #     ['emb_s_a.weight', 'emb_x_a.weight', 'emb_y_a.weight', 'emb_z_a.weight', 'rel_s_b.weight', 'rel_x_b.weight',
        #      'rel_y_b.weight', 'rel_z_b.weight', 'rel_w.weight', 'fc.weight', 'bn.weight', 'bn.bias', 'bn.running_mean',
        #      'bn.running_var', 'bn.num_batches_tracked'])
        ent_embeddings = data['emb_s_a.weight']
        rel_embeddings = data['rel_s_b.weight']

    with open(dataset + '/entity2vec.QuatE', 'w', encoding='utf-8') as f:
        for emb in ent_embeddings:
            for idx, e in enumerate(list(emb)):
                if idx == len(list(emb))-1:
                    f.write(str(e)+'\n')
                else:
                    f.write(str(e)+'\t')

    with open(dataset + '/relation2vec.QuatE', 'w', encoding='utf-8') as f:
        for emb in rel_embeddings:
            for idx, e in enumerate(list(emb)):
                if idx == len(list(emb)) - 1:
                    f.write(str(e) + '\n')
                else:
                    f.write(str(e) + '\t')

def convert_transe_vec(dataset):
    with open(dataset + '/transe.parameter.json') as f:
        data = json.load(f)
        # dict_keys(['zero_const', 'pi_const', 'ent_embeddings.weight', 'rel_embeddings.weight'])
        ent_embeddings = data['ent_embeddings.weight']
        rel_embeddings = data['rel_embeddings.weight']

    with open(dataset + '/entity2vec.TransE', 'w', encoding='utf-8') as f:
        for emb in ent_embeddings:
            for idx, e in enumerate(list(emb)):
                if idx == len(list(emb))-1:
                    f.write(str(e)+'\n')
                else:
                    f.write(str(e)+'\t')

    with open(dataset + '/relation2vec.TransE', 'w', encoding='utf-8') as f:
        for emb in rel_embeddings:
            for idx, e in enumerate(list(emb)):
                if idx == len(list(emb)) - 1:
                    f.write(str(e) + '\n')
                else:
                    f.write(str(e) + '\t')

def tail2candidate(dataset):
    head_dict = {}
    relations = [key for key in (json.load(open(dataset + '/train_tasks.json'))).keys() if '_inv' not in key]

    with open(dataset + '/path_graph') as f:
        for line in f:
            line = line.rstrip()
            e1 = line.split('\t')[0]
            rel = line.split('\t')[1]
            if 'inv' in rel:
                continue
            e2 = line.split('\t')[2]
            if e1 in head_dict:
                head_dict[e1] += [[e1, rel, e2]]
            else:
                head_dict[e1] = [[e1, rel, e2]]
    test_tasks = json.load(open(dataset + '/test_tasks_query.json'))

    all_reason_head = []
    for data in list(test_tasks.values()):
        for triple in data:
            all_reason_head.append(triple[0])
    head2candidates = {}
    for idx, head in enumerate(all_reason_head):
        candidates = []
        if head in head_dict:
            # 一跳
            fil_o_data = [data[2] for data in head_dict[head]]
            # fil_data = [data for data in head_dict[head]]
            candidates += fil_o_data
            # print('head', head)
            # print('1st fil_o_data', fil_o_data)
            # print('1st fil_data', fil_data)
            for tail in fil_o_data:
                if tail not in head_dict:
                    continue
                fil_o_data = [data[2] for data in head_dict[tail]]
                # fil_data = [data for data in head_dict[tail]]
                candidates += fil_o_data
                # print('2nd fil_o_data', fil_o_data)
                # print('2nd fil_data', fil_data)
                for tail in fil_o_data:
                    if tail not in head_dict:
                        continue
                    fil_o_data = [data[2] for data in head_dict[tail]]
                    # fil_data = [data for data in head_dict[tail]]
                    candidates += fil_o_data
                    # print('3rd fil_o_data', fil_o_data)
                    # print('3rd fil_data', fil_data)
                    for tail in fil_o_data:
                        if tail not in head_dict:
                            continue
                        fil_o_data = [data[2] for data in head_dict[tail]]
                        # fil_data = [data for data in head_dict[tail]]
                        candidates += fil_o_data
                        # print('4th fil_o_data', fil_o_data)
                        # print('4th fil_data', fil_data)
                        for tail in fil_o_data:
                            if tail not in head_dict:
                                continue
                            fil_o_data = [data[2] for data in head_dict[tail]]
                            # fil_data = [data for data in head_dict[tail]]
                            candidates += fil_o_data
                            # print('5th fil_o_data', fil_o_data)
                            # print('5th fil_data', fil_data)
                            for tail in fil_o_data:
                                if tail not in head_dict:
                                    continue
                                fil_o_data = [data[2] for data in head_dict[tail]]
                                # fil_data = [data for data in head_dict[tail]]
                                candidates += fil_o_data
                                # print('6th fil_o_data', fil_o_data)
                                # print('6th fil_data', fil_data)

        head2candidates[head] = list(set(candidates))
        print(idx, head, len(head2candidates[head])) #head2candidates[head])

    json.dump(head2candidates, open(dataset + '/head2candidates.json', 'w'), ensure_ascii=False, indent=4)


def test(dataset):
    with open(dataset + '/head2candidates.json', 'r') as file:
        head2candidates = json.load(file)
    with open(dataset + '/test_tasks_query.json', 'r') as file:
        test_tasks_query = json.load(file)
    test_pairs = []
    for key in test_tasks_query.keys():
        for d in test_tasks_query[key]:
            test_pairs.append(d)
    print(len(test_pairs))

    for pair in test_pairs:
        head = pair[0]
        tail = pair[2]
        if tail not in head2candidates[head]:
            print(pair)

if __name__ == '__main__':
    start = time()

    PRETRAIN_TYPE = 'bid_v1'
    if PRETRAIN_TYPE == 'TransE':
        #build ent2ids rel2ids
        DATASET = './ARMY_TransE_v3'
        build_vocab(DATASET)
        # build e1rel_e2.json
        for_filtering(DATASET, save=True)
        # rel2candidates.json
        candidate_triples(DATASET)
        # Convert Embedding
        convert_vec(DATASET)
        print('Time eclipse: ', time() - start)
        # tail2candidate(DATASET)
        # test(DATASET)
    elif PRETRAIN_TYPE == 'QuatE':
        #build ent2ids rel2ids
        DATASET = './ARMY_QuatE'
        build_vocab(DATASET)
        # build e1rel_e2.json
        for_filtering(DATASET, save=True)
        # rel2candidates.json
        candidate_triples(DATASET)
        # Convert Embedding
        convert_quate_vec(DATASET)
        print('Time eclipse: ', time() - start)
        tail2candidate(DATASET)
        test(DATASET)
    elif PRETRAIN_TYPE == 'QuatE_v2':
        #build ent2ids rel2ids
        DATASET = './ARMY_QuatE_v2'
        build_vocab(DATASET)
        # build e1rel_e2.json
        for_filtering(DATASET, save=True)
        # rel2candidates.json
        candidate_triples(DATASET)
        # Convert Embedding
        convert_quate_vec(DATASET)
        print('Time eclipse: ', time() - start)
        tail2candidate(DATASET)
        test(DATASET)
    elif PRETRAIN_TYPE == 'QuatE_v3':
        #build ent2ids rel2ids
        DATASET = './ARMY_QuatE_v3'
        build_vocab(DATASET)
        # build e1rel_e2.json
        for_filtering(DATASET, save=True)
        # rel2candidates.json
        candidate_triples(DATASET)
        # Convert Embedding
        convert_quate_vec(DATASET)
        print('Time eclipse: ', time() - start)
        tail2candidate(DATASET)
        test(DATASET)
    elif PRETRAIN_TYPE == 'bid_prediction_match':
        DATASET = './bid_prediction_match'
        build_vocab(DATASET)
        convert_transe_vec(DATASET)
        print('Time eclipse: ', time() - start)
    elif PRETRAIN_TYPE == 'bid_v1':
        DATASET = './bid_v1'
        build_vocab(DATASET)
        convert_transe_vec(DATASET)
        print('Time eclipse: ', time() - start)


