
import math
#拼接（3*64）卷积
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
class DenoisingAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_level):
        super(DenoisingAutoEncoder, self).__init__()
        self.noise_level = noise_level
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # Sigmoid 函数用于将输出限制在 [0, 1] 范围内
        )
        self._init_weight2()

    def forward(self, x):
        noisy_x = x + self.noise_level * torch.randn_like(x)  # 在输入数据上添加高斯噪声
        encoded = self.encoder(noisy_x)
        decoded = self.decoder(encoded)
        return decoded
    def _init_weight2(self):

        for layer in self.encoder:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.decoder:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)


class CKAN(nn.Module):
    def __init__(self, args, n_entity, n_relation,emb_noise_level=0.3):
        super(CKAN, self).__init__()
        self._parse_args(args, n_entity, n_relation)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.scaler = StandardScaler()
        self.attention = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
                )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #----------------------------------------------------------------------------------------------
        self.user_dae = DenoisingAutoEncoder(input_dim=self.dim, hidden_dim=self.dim//2, noise_level=emb_noise_level)
        self.item_dae = DenoisingAutoEncoder(input_dim=self.dim, hidden_dim=self.dim//2, noise_level=emb_noise_level)


        self.W_Q_U=[]
        for i in range(0,self.n_layer):
            x=nn.Parameter(torch.randn(self.dim, self.dim)).to(device)
            self.W_Q_U.append(x)

        self.W_K_U = []
        for i in range(0, self.n_layer):
            x = nn.Parameter(torch.randn(self.dim, self.dim)).to(device)
            self.W_K_U.append(x)

        self.W_Q_V=[]
        for i in range(0,self.n_layer):
            x=nn.Parameter(torch.randn(self.dim, self.dim)).to(device)
            self.W_Q_V.append(x)


        self.W_K_V = []
        for i in range(0, self.n_layer):
            x = nn.Parameter(torch.randn(self.dim, self.dim)).to(device)
            self.W_K_V.append(x)

        self.ConVU = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(self.n_layer+1, self.n_layer+1))
        self.ConVV = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(self.n_layer+1, self.n_layer+1))
        #----------------------------------------------------------------------------------------------
        self._init_weight()
    def forward(
        self,
        items: torch.LongTensor,
        user_triple_set: list,
        item_triple_set: list,
    ):
        user_embeddings = []

        # [batch_size, triple_set_size, dim]
        user_emb_0 = self.entity_emb(user_triple_set[0][0])
        noisy_user_emb_0 = self.user_dae(user_emb_0.mean(dim=1))  # 应用 DAE
        user_embeddings.append(noisy_user_emb_0)

        for i in range(self.n_layer):

            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(user_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(user_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_triple_set[2][i])
            # [batch_size, dim]
            user_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            e_u_Q = torch.matmul(user_emb_0.mean(dim=1), self.W_Q_U[i])
            e_u_K = torch.matmul(user_emb_i, self.W_K_U[i])
            su = (e_u_Q * e_u_K).sum(dim=1)
            su = su / math.sqrt(self.dim)
            su = su.unsqueeze(1)
            su = F.softmax(su, dim=-1)
            user_emb_i = user_emb_i * su + user_emb_i
            user_embeddings.append(user_emb_i)

        item_embeddings = []

        # [batch size, dim]
        item_emb_origin = self.entity_emb(items)
        noisy_item_emb_origin = self.item_dae(item_emb_origin)  # 应用 DAE
        item_embeddings.append(item_emb_origin)

        # item_emb_0 = self.entity_emb(item_triple_set[0][0])
        # # [batch_size, dim]
        # noisy_item_emb_origin = self.item_dae(item_emb_0.mean(dim=1))
        # item_embeddings.append(noisy_item_emb_origin)

        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(item_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(item_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_triple_set[2][i])
            # [batch_size, dim]
            item_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            e_v_Q = torch.matmul(noisy_item_emb_origin, self.W_Q_V[i])
            e_v_K = torch.matmul(item_emb_i, self.W_K_V[i])
            sv = (e_v_Q * e_v_K).sum(dim=1)
            sv = sv / math.sqrt(self.dim)
            sv = sv.unsqueeze(1)
            sv = F.softmax(sv, dim=-1)
            item_emb_i = item_emb_i*sv
            item_embeddings.append(item_emb_i)

        if self.n_layer > 0 and (self.agg == 'sum' or self.agg == 'pool'):
              # [batch_size, triple_set_size, dim]
              pass
        # item_emb_0 = self.entity_emb(item_triple_set[0][0])
        #     # [batch_size, dim]
        # item_embeddings.append(item_emb_0.mean(dim=1))

        scores = self.predict2(user_embeddings, item_embeddings)

        # DAE Loss
        user_emb_reconstructed = user_emb_0.mean(dim=1)
        user_dae_loss = F.mse_loss(user_emb_reconstructed, noisy_user_emb_0)

        item_emb_reconstructed = self.entity_emb(items)
        item_dae_loss = F.mse_loss(item_emb_reconstructed, noisy_item_emb_origin)
        return scores,user_dae_loss,item_dae_loss

    def predict2(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
        e_u = torch.unsqueeze(e_u, 1)
        e_v = torch.unsqueeze(e_v, 1)

        for i in range(1, len(user_embeddings)):
            e = torch.unsqueeze(user_embeddings[i], 1)
            e_u = torch.cat((e, e_u), dim=1)

        for i in range(1, len(item_embeddings)):
            e = torch.unsqueeze(item_embeddings[i], 1)
            e_v = torch.cat((e, e_v), dim=1)

        e_u=torch.unsqueeze(e_u, 1)
        e_v=torch.unsqueeze(e_v, 1)
        #torch.Size([2048, 1, 4, 64])

        u=self.ConVU(e_u)
        v=self.ConVV(e_v)

        u = torch.squeeze(u, 1)
        v=torch.squeeze(v, 1)
        u = torch.squeeze(u, 1)
        v = torch.squeeze(v, 1)


        scores = (v * u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores
    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]

        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u),dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v),dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):
                e_u += user_embeddings[i]
            for i in range(1, len(item_embeddings)):
                e_v += item_embeddings[i]
        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v, item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)

        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores


    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg


    def _init_weight(self):
        # init embedding
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        #----------------------------------------------------------------------
        nn.init.xavier_uniform_(self.ConVU.weight)
        nn.init.xavier_uniform_(self.ConVV.weight)
        # init attention
        for layer in self.attention:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)




    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        # [batch_size, triple_set_size]
        att_weights = self.attention(torch.cat((h_emb,r_emb),dim=-1)).squeeze(-1)
        # [batch_size, triple_set_size]
        att_weights_norm = F.softmax(att_weights,dim=-1)
        # [batch_size, triple_set_size, dim]
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        # [batch_size, dim]
        emb_i = emb_i.sum(dim=1)
        return emb_i

