from encoder import UserEncoder, TextEncoder
import torch.optim as optim
from torch import nn
import torch 
import numpy as np
from dataset import NewsDataset, NewsPartDataset, NewsUpdatorDataset
from torch.utils.data import DataLoader


class UserModel(nn.Module):
    def __init__(self, news_dataset, news_index, 
                 device, news_embedding_dim=400,
                 num_attention_heads=20,
                 query_vector_dim=200):
        
        super(UserModel, self).__init__()
        self.device = device

        self.text_encoder = TextEncoder().to(device)
        self.user_encoder = UserEncoder().to(device)

        self.news_optimizer = optim.Adam(self.text_encoder.parameters(), lr=0.00005)
        self.user_optimizer = optim.Adam(self.user_encoder.parameters(), lr=0.00005)

        for param in self.text_encoder.DistillBert.parameters():
            param.requires_grad = False
        
        self.news_dataset = news_dataset
        self.news_index = news_index

        self.time = 0
        self.cnt = 0
        self.criterion = nn.CrossEntropyLoss()
        self._init_grad_param()

    def _init_grad_param(self):
        self.news_grads = {}
        self.user_optimizer.zero_grad()
        self.news_optimizer.zero_grad()

    def gen_news_vecs(self, nids):
        self.text_encoder.eval()
        news_vectors = []
        for i in nids:
            news_ds = NewsPartDataset(self.news_index, i)
            news_dl = DataLoader(news_ds, batch_size=2048, shuffle=False, num_workers=0)
            """
            A dataloader that containers nid, news_index
            """
            # news_vecs = np.zeros((npratio, 400), dtype='float32')
            news_vecs = []
            with torch.no_grad():
                for nids, news in news_dl:
                    news = news.to(self.device)
                    news_vec = self.text_encoder(news).detach().cpu().numpy()
                    # news_vecs[nids.numpy()] = news_vec
                    news_vecs=news_vec
            if np.isnan(news_vecs).any():
                raise ValueError("news_vecs contains nan")
            news_vectors.append(news_vecs)
        return news_vectors

    def get_news_vecs(self, idx):
        return self.news_vecs[idx]

    def update(self):
        self.update_user_grad()
        self.update_news_grad()
        self._init_grad_param()
        self.cnt += 1

    def update_news_grad(self):
        self.text_encoder.train()
        self.news_optimizer.zero_grad()

        news_ids, news_grads = [], []
        for nid in self.news_grads:
            news_ids.append(nid)
            news_grads.append(self.news_grads[nid])

        news_up_ds = NewsUpdatorDataset(self.news_index, news_ids, news_grads)
        news_up_dl = DataLoader(news_up_ds, batch_size=128, shuffle=False, num_workers=0)
        for news_index, news_grad in news_up_dl:
            news_index = news_index.to(self.device)
            news_grad = news_grad.to(self.device)
            news_vecs = self.text_encoder(news_index)
            news_vecs.backward(news_grad)

        self.news_optimizer.step()
        self.news_optimizer.zero_grad()

    def update_user_grad(self):
        self.user_optimizer.step()
        self.user_optimizer.zero_grad()


    def collect(self, news_grad, user_grad):
        # update user model params
        for name, param in self.user_encoder.named_parameters():
            if param.grad is None:
                param.grad = user_grad[name]
            else:
                param.grad += user_grad[name]
        # update news model params
        for nid in news_grad:
            if nid in self.news_grads:
                self.news_grads[nid] += news_grad[nid]
            else:
                self.news_grads[nid] = news_grad[nid]

    def forward(self, candidate_news, his_news, labels, compute_loss=True):
        candidate_vecs = torch.as_tensor(np.array(self.gen_news_vecs(candidate_news))).to(self.device)

        his_vec = torch.as_tensor(np.array(self.gen_news_vecs(his_news))).to(self.device)

        candidate_vecs.requires_grad = True
        his_vec.requires_grad = True

        user_vector = self.user_encoder(his_vec)    

        score = torch.bmm(candidate_vecs, user_vector.unsqueeze(-1)).squeeze(dim=-1)
       
        score = torch.sigmoid(score)

        if compute_loss:
            loss = self.criterion(score, labels)
            return loss, score, candidate_vecs, his_vec
        else:
            return score, candidate_vecs, his_vec
        



class ServerUserModel(nn.Module):
    def __init__(self, device):
        
        super(UserModel, self).__init__()
        self.device = device

        self.text_encoder = TextEncoder().to(device)
        self.user_encoder = UserEncoder().to(device)

        self.news_optimizer = optim.Adam(self.text_encoder.parameters(), lr=0.00005)
        self.user_optimizer = optim.Adam(self.user_encoder.parameters(), lr=0.00005)

        for param in self.text_encoder.DistillBert.parameters():
            param.requires_grad = False
        

        self.time = 0
        self.cnt = 0
        self.criterion = nn.CrossEntropyLoss()
        self._init_grad_param()

    def _init_grad_param(self):
        self.news_grads = {}
        self.user_optimizer.zero_grad()
        self.news_optimizer.zero_grad()

    def gen_news_vecs(self, nids):
        self.text_encoder.eval()
        news_vectors = []
        for i in nids:
            news_ds = NewsPartDataset(self.news_index, i)
            news_dl = DataLoader(news_ds, batch_size=2048, shuffle=False, num_workers=0)
            """
            A dataloader that containers nid, news_index
            """
            # news_vecs = np.zeros((npratio, 400), dtype='float32')
            news_vecs = []
            with torch.no_grad():
                for nids, news in news_dl:
                    news = news.to(self.device)
                    news_vec = self.text_encoder(news).detach().cpu().numpy()
                    # news_vecs[nids.numpy()] = news_vec
                    news_vecs=news_vec
            if np.isnan(news_vecs).any():
                raise ValueError("news_vecs contains nan")
            news_vectors.append(news_vecs)
        return news_vectors

    def get_news_vecs(self, idx):
        return self.news_vecs[idx]

    def update(self):
        self.update_user_grad()
        self.update_news_grad()
        self._init_grad_param()
        self.cnt += 1

    def update_news_grad(self):
        self.text_encoder.train()
        self.news_optimizer.zero_grad()

        news_ids, news_grads = [], []
        for nid in self.news_grads:
            news_ids.append(nid)
            news_grads.append(self.news_grads[nid])

        news_up_ds = NewsUpdatorDataset(self.news_index, news_ids, news_grads)
        news_up_dl = DataLoader(news_up_ds, batch_size=128, shuffle=False, num_workers=0)
        for news_index, news_grad in news_up_dl:
            news_index = news_index.to(self.device)
            news_grad = news_grad.to(self.device)
            news_vecs = self.text_encoder(news_index)
            news_vecs.backward(news_grad)

        self.news_optimizer.step()
        self.news_optimizer.zero_grad()

    def update_user_grad(self):
        self.user_optimizer.step()
        self.user_optimizer.zero_grad()


    def collect(self, news_grad, user_grad):
        # update user model params
        for name, param in self.user_encoder.named_parameters():
            if param.grad is None:
                param.grad = user_grad[name]
            else:
                param.grad += user_grad[name]
        # update news model params
        for nid in news_grad:
            if nid in self.news_grads:
                self.news_grads[nid] += news_grad[nid]
            else:
                self.news_grads[nid] = news_grad[nid]

    def forward(self, candidate_news, his_news, labels, compute_loss=True):
        candidate_vecs = torch.as_tensor(np.array(self.gen_news_vecs(candidate_news))).to(self.device)

        his_vec = torch.as_tensor(np.array(self.gen_news_vecs(his_news))).to(self.device)

        candidate_vecs.requires_grad = True
        his_vec.requires_grad = True

        user_vector = self.user_encoder(his_vec)    

        score = torch.bmm(candidate_vecs, user_vector.unsqueeze(-1)).squeeze(dim=-1)
       
        score = torch.sigmoid(score)

        if compute_loss:
            loss = self.criterion(score, labels)
            return loss, score, candidate_vecs, his_vec
        else:
            return score, candidate_vecs, his_vec