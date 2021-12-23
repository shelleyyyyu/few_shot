import json
import logging
import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from args import read_options
from data_loader import *
from matcher import *
from tensorboardX import SummaryWriter

class Trainer(object):
    
    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items(): setattr(self, k, v)

        self.meta = not self.no_meta

        if self.random_embed:
            use_pretrain = False
        else:
            use_pretrain = True

        logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        if self.test or self.random_embed:
            self.load_symbol2id()
            use_pretrain = False
        else:
            # load pretrained embedding
            self.load_embed()
        self.use_pretrain = use_pretrain
        self.num_symbols = len(self.symbol2id.keys()) - 1 # one for 'PAD'
        self.pad_id = self.num_symbols
        self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols, use_pretrain=self.use_pretrain, embed=self.symbol2vec, dropout=self.dropout, batch_size=self.batch_size, process_steps=self.process_steps, finetune=self.fine_tune, aggregate=self.aggregate, padid=self.pad_id)
        if torch.cuda.is_available():
            self.matcher.cuda()

        self.batch_nums = 0
        if self.test:
            self.writer = None
        else:
            self.writer = SummaryWriter('logs/' + self.prefix)

        self.parameters = filter(lambda p: p.requires_grad, self.matcher.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[200000], gamma=0.5)

        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.rel2id = json.load(open(self.dataset + '/relation2ids'))
        self.num_ents = len(self.ent2id.keys())

        # logging.info('BUILDING CONNECTION MATRIX')
        # degrees = self.build_connection(max_=self.max_neighbor)

        # logging.info('LOADING CANDIDATES ENTITIES')
        # self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))
        # self.head2candidates = json.load(open(self.dataset + '/head2candidates.json'))


        # # load answer dict
        # self.e1rel_e2 = defaultdict(list)
        # self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

    def load_symbol2id(self):
        
        symbol_id = {}
        self.rel2id = json.load(open(self.dataset + '/relation2ids'))
        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        i = 0
        for key in self.rel2id.keys():
            if key not in ['','OOV']:
                symbol_id[key] = i
                i += 1

        for key in self.ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

    def load_embed(self):

        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))

        logging.info('LOADING PRE-TRAINED EMBEDDING')
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL', 'QuatE']:
            ent_embed = np.loadtxt(self.dataset + '/entity2vec.' + self.embed_model)
            rel_embed = np.loadtxt(self.dataset + '/relation2vec.' + self.embed_model)

            if self.embed_model == 'ComplEx' or self.embed_model == 'QuatE':
                # normalize the complex embeddings
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            assert ent_embed.shape[0] == len(ent2id.keys())
            assert rel_embed.shape[0] == len(rel2id.keys())

            i = 0
            embeddings = []
            for key in rel2id.keys():
                if key not in ['','OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(rel_embed[rel2id[key],:]))

            for key in ent2id.keys():
                if key not in ['', 'OOV']:
                    if key in symbol_id:
                        print('duplicated entity: ' + key)
                        continue
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(ent_embed[ent2id[key],:]))

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)
            assert embeddings.shape[0] == len(symbol_id.keys())

            self.symbol2id = symbol_id
            self.symbol2vec = embeddings

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.matcher.state_dict(), path)

    def load(self):
        self.matcher.load_state_dict(torch.load(self.save_path))

    def train(self):
        logging.info('START TRAINING...')

        best_accuracy = 0.0

        losses = deque([], self.log_every)
        margins = deque([], self.log_every)

        for data in train_generate_matcher(self.dataset, self.batch_size, self.symbol2id):
            support_pairs, query_pairs, false_pairs = data
            # Support 應該是機構以及其相關特徵
            # Query 是項目
            # False 是未中標項目
            query_scores = self.matcher(query_pairs, support_pairs)
            false_scores = self.matcher(query_pairs, false_pairs)
            # print('query_scores', query_scores)
            # print('false_scores', false_scores)
            margin_ = query_scores - false_scores
            # print('margin_', margin_)
            margins.append(margin_.mean().item())
            # print('margins', margins)
            loss = F.relu(self.margin - margin_).mean()
            # print('loss', loss)
            # exit()
            self.writer.add_scalar('MARGIN', np.mean(margins), self.batch_nums)
            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(self.parameters, self.grad_clip)
            self.optim.step()

            if self.batch_nums != 0 and self.batch_nums % self.eval_every == 0:
                accuracy, _ = self.eval()
                self.writer.add_scalar('Accuracy', float(accuracy), self.batch_nums)
                self.save()

                if accuracy > best_accuracy:
                    self.save(self.save_path + '_bestAccuracy')
                    best_accuracy = accuracy

            if self.batch_nums % self.log_every == 0:
                logging.info('AVG. BATCH_LOSS: %.4f AT STEP %d'%(float(np.mean(losses)), self.batch_nums))
                self.writer.add_scalar('Avg_batch_loss', np.mean(losses), self.batch_nums)


            self.batch_nums += 1
            self.scheduler.step()
            if self.batch_nums == self.max_batches:
                self.save()
                break

    def eval(self, mode='test'):
        #TODO- remove the testing setting
        self.matcher.eval()

        symbol2id = self.symbol2id
        logging.info('EVALUATING ON %s DATA' % mode.upper())
        test_tasks = json.load(open(self.dataset + '/test_tasks.json'))
        result = {}
        total_correct_count = 0
        for query_ in test_tasks.keys():

            all_test_triples = test_tasks[query_]
            query_pairs = [symbol2id[query_]]*len(all_test_triples)

            support_pairs = []
            ground_truth_list = []
            scores_list = []

            for company, feature in all_test_triples.items():
                support_pair = [[symbol2id[relation], symbol2id[tail_entity]] for relation, tail_entity in feature.items() if relation != '投标_项目_公司_是否_中标' and relation != '投标_项目_公司_本次_评标_排名']
                support_pair.append([symbol2id['投标_公司'], symbol2id[company]])
                support_pair.append([symbol2id['投标_项目'], symbol2id[query_]])
                support_pairs.append(support_pair)
                ground_truth_list.append([feature['投标_项目_公司_是否_中标'], feature['投标_项目_公司_本次_评标_排名']])

            for i in range(len(query_pairs)):
                score = self.matcher([query_pairs[i]], [support_pairs[i]])
                if torch.cuda.is_available():
                    score = score.detach().cpu().numpy()
                else:
                    score = score.detach().numpy()
                scores_list.append(score)

            sort_scores_list = list(np.argsort(scores_list))[::-1]
            bid_sucess_id = -1
            for idx, ground_truth in enumerate(ground_truth_list):
                if ground_truth[0] == 'true':
                    bid_sucess_id = idx
                    break
            if sort_scores_list[0] == bid_sucess_id:
                total_correct_count += 1

            tmp_dict = {}
            tmp_dict['prediction'] = scores_list
            tmp_dict['ground_truth'] = ground_truth_list
            result[query_] = tmp_dict

        accuracy = float(total_correct_count/len(list(test_tasks.keys())))
        logging.critical('Top Prediction Accuracy: {}'.format('%.4f'%(accuracy)))
        logging.info('Number of text examples {}'.format(len(list(test_tasks.keys()))))

        self.matcher.train()
        return accuracy, result

    def test_(self):
        self.load()
        logging.info('Pre-trained model loaded')
        self.eval(mode='test', meta=self.meta)

if __name__ == '__main__':
    args = read_options()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    # setup random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args)
    if args.test:
        trainer.test_()
    else:
        trainer.train()
