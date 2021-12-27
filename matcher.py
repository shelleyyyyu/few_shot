import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from modules import *
from torch.autograd import Variable

class EmbedMatcher(nn.Module):
    """
    Matching metric based on KB Embeddings
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max', padid='-1'):
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
        self.aggregate = aggregate
        self.num_symbols = num_symbols
        self.padid = padid
        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.dropout = nn.Dropout(0.5)

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)

        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        d_model = self.embed_dim*2
        self.support_encoder = SupportEncoder(self.embed_dim, 2*self.embed_dim, dropout)
        self.query_encoder = QueryEncoder(self.embed_dim, process_steps)


    def neighbor_encoder(self, connections, num_neighbors):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        relations = []
        for connection in connections:
            tmp_array = []
            for c in connection:
                tmp_array.append(c[0])
            if len(tmp_array) < 5:
                tmp_array = tmp_array + [self.padid]*(5-len(tmp_array))
            relations.append(tmp_array)
        relations = Variable(torch.LongTensor(np.stack(relations, axis=0)))
        if torch.cuda.is_available():
            relations = relations.cuda()
        rel_embeds = self.dropout(self.symbol_emb(relations)) # (batch, 200, embed_dim)

        entities = []
        for connection in connections:
            tmp_array = []
            for c in connection:
                tmp_array.append(c[1])
            if len(tmp_array) < 5:
                tmp_array = tmp_array + [self.padid]*(5-len(tmp_array))
            entities.append(tmp_array)
        entities = Variable(torch.LongTensor(np.stack(entities, axis=0)))
        if torch.cuda.is_available():
            entities = entities.cuda()

        ent_embeds = self.dropout(self.symbol_emb(entities))  # (batch, 200, embed_dim)
        rel_embeds = self.dropout(rel_embeds) # (batch, 200, embed_dim)
        ent_embeds = self.dropout(ent_embeds) # (batch, 200, embed_dim)
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1) # (batch, 200, 2*embed_dim)
        out = self.gcn_w(concat_embeds) # (batch, embed_dim)
        out = torch.sum(out, dim=1) # (batch, embed_dim)
        out = out / num_neighbors # (batch, embed_dim)
        return out.tanh()

    #def forward(self, query, support, query_meta=None, support_meta=None):
    def forward(self, query_pairs, support_pairs):
        '''
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''
        support_neighbor = self.neighbor_encoder(support_pairs, len(support_pairs))
        support = support_neighbor # batch_size * 200
        support_g = self.support_encoder(support) # batch_size * 200
        query = Variable(torch.LongTensor(np.stack(query_pairs, axis=0))) #batch_size
        if torch.cuda.is_available():
            query = query.cuda()
        query_embeds = self.dropout(self.symbol_emb(query))  # (batch, 200, embed_dim)
        # print(support_g.size())
        # print(query_embeds.size())
        query_embeds = self.query_encoder(support_g, query_embeds)
        # print(query_embeds.size())
        # exit()
        mean_support_g = torch.mean(support_g, dim=1)
        mean_query_g = torch.mean(query_embeds, dim=1)
        matching_scores = torch.matmul(mean_query_g, mean_support_g).squeeze()
        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # matching_scores = cos(query_embeds, support_g)
        # print(matching_scores)
        return matching_scores

if __name__ == '__main__':

    query = Variable(torch.ones(64,2).long()) * 100
    support = Variable(torch.ones(40,2).long())
    matcher = EmbedMatcher(40, 200, use_pretrain=False)
    print(matcher(query, support).size())


