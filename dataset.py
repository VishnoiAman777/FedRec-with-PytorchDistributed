from torch.utils.data import Dataset
import numpy as np
import random

"""
This function is just to adjust the ratio of positive and negative classes
"""
npratio = 4
max_his_len=50
def newsample(nnn, ratio):
    if ratio > len(nnn):
        return nnn + ["<unk>"] * (ratio - len(nnn))
    else:
        return random.sample(nnn, ratio)




"""
This class just returns distil_bert title and it's curresponding mask for a particular news
"""
class NewsDataset(Dataset):
    def __init__(self, news_index):
        self.news_index = news_index

    def __len__(self):
        return len(self.news_index)

    def __getitem__(self, idx):
        return self.news_index[idx]
    

"""
Return <token, attention_mask> curresponding to a particular NewsID
"""    
class NewsPartDataset(Dataset):
    def __init__(self, news_index, nids):
        self.news_index = news_index
        self.nids = nids

    def __len__(self):
        return len(self.nids)

    def __getitem__(self, idx):
        nid = self.nids[idx]
        return nid, self.news_index[nid]
    
"""
This is used for updating news at a  particular index
Return [News_Embeddings, Attention_mask] and it's gradients
"""
class NewsUpdatorDataset(Dataset):
    def __init__(self, news_index, news_ids, news_grads):
        self.news_index = news_index
        self.news_grads = news_grads
        self.news_ids = news_ids

    def __len__(self):
        return len(self.news_ids)

    def __getitem__(self, idx):
        nid = self.news_ids[idx]
        return self.news_index[nid], self.news_grads[idx]
    

"""

"""
class TrainDataset(Dataset):
    def __init__(self, train_sam, nid2index, news_index):
        # Here samples is same as train_sam_uid stuff 
        self.news_index = news_index
        self.nid2index = nid2index
        self.train_sam = train_sam

    def __len__(self):
        return len(self.train_sam)

    def __getitem__(self, idx):
        # pos, neg, his, neg_his
        _, pos, neg, his, _ = self.train_sam[idx]
        neg = newsample(neg, npratio)
        candidate_news = np.array([self.nid2index[n] for n in [pos] + neg])
        his = np.array([self.nid2index[n] for n in his] + [0] * (max_his_len - len(his)))
        label = np.array(0)
        return candidate_news, his, label
    

"""
    This class returns history for  user 
"""
class UserDataset(Dataset):
    def __init__(self,
                 args,
                 samples,
                 news_vecs,
                 nid2index):
        self.samples = samples
        self.args = args
        self.news_vecs = news_vecs
        self.nid2index = nid2index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, poss, negs, his, _ = self.samples[idx]
        his = [self.nid2index[n] for n in his] + [0] * (self.args.max_his_len - len(his))
        his = self.news_vecs[his]
        return his