import torch
from torch import nn
from transformers import DistilBertModel
from attention import AdditiveAttention
from attention import MultiHeadAttention
import torch.nn.functional as F


"""
This class contains the text Encoder DistillBert + Additive Attention which will return the textual Embedding of a particular News
"""
class TextEncoder(nn.Module):
    def __init__(self,
                 word_embedding_dim=400,
                 dropout_rate=0.2,
                 enable_gpu=True):
        super(TextEncoder, self).__init__()
        self.dropout_rate = 0.2
        self.DistillBert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.additive_attention = AdditiveAttention(self.DistillBert.config.hidden_size,
                                                    self.DistillBert.config.hidden_size // 2)
        self.fc = nn.Linear(self.DistillBert.config.hidden_size, word_embedding_dim)

    def forward(self, text):
        tokens = text[:, 0, :]
        atts = text[:, 1, :]
        text_vector = self.DistillBert(tokens, attention_mask=atts)[0]
        text_vector = self.additive_attention(text_vector)
        text_vector = self.fc(text_vector)
        return text_vector
    

"""
This encodes the userHistory by using MultiHead Attention + Additive Attention Module
"""
class UserEncoder(nn.Module):
    def __init__(self,
                 news_embedding_dim=400,
                 num_attention_heads=20,
                 query_vector_dim=200
                 ):
        super(UserEncoder, self).__init__()
        self.dropout_rate = 0.2
        self.multihead_attention = MultiHeadAttention(news_embedding_dim,
                                                      num_attention_heads, 20, 20)
        self.additive_attention = AdditiveAttention(news_embedding_dim,
                                                    query_vector_dim)

    def forward(self, clicked_news_vecs):
        clicked_news_vecs = F.dropout(clicked_news_vecs, p=self.dropout_rate, training=self.training)
        multi_clicked_vectors = self.multihead_attention(
            clicked_news_vecs, clicked_news_vecs, clicked_news_vecs
        )
        pos_user_vector = self.additive_attention(multi_clicked_vectors)
        user_vector = pos_user_vector
        return user_vector
