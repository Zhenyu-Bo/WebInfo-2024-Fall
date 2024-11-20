# models.py

import torch
from torch import nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim, user_tag_embedding_dim, book_tag_embedding_dim, 
                 use_tags=True, dropout_rate=0.5):
        super(MatrixFactorization, self).__init__()
        self.use_tags = use_tags
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.book_embeddings = nn.Embedding(num_books, embedding_dim)

        self.user_tag_linear = nn.Linear(user_tag_embedding_dim, embedding_dim) if self.use_tags else None
        self.book_tag_linear = nn.Linear(book_tag_embedding_dim, embedding_dim) if self.use_tags else None

        # 时间特征嵌入
        self.year_embeddings = nn.Embedding(20, 10)
        self.month_embeddings = nn.Embedding(13, 5)
        self.day_embeddings = nn.Embedding(32, 5)
        self.hour_embeddings = nn.Embedding(24, 5)
        self.weekday_embeddings = nn.Embedding(7, 5)
        self.isweekend_embeddings = nn.Embedding(2, 2)

        self.dropout = nn.Dropout(p=dropout_rate)
        # 根据实际的拼接方式调整interaction_dim
        interaction_dim = 64 + 64 + 64 + 32  # combined_user*book(64) + combined_user(64) + combined_book(64) + time_emb(32) = 224
        self.output = nn.Linear(interaction_dim, 1)

    def forward(self, user, book, user_tag_embedding, book_tag_embedding, time_features):
        user_emb = self.user_embeddings(user)
        book_emb = self.book_embeddings(book)

        if self.use_tags and self.user_tag_linear:
            user_tag_emb_proj = self.user_tag_linear(user_tag_embedding)
            combined_user = torch.cat([user_emb, user_tag_emb_proj], dim=1)  # 64
        else:
            combined_user = user_emb  # 32

        if self.use_tags and self.book_tag_linear:
            book_tag_emb_proj = self.book_tag_linear(book_tag_embedding)
            combined_book = torch.cat([book_emb, book_tag_emb_proj], dim=1)  # 64
        else:
            combined_book = book_emb  # 32

        year, month, day, hour, weekday, isweekend = time_features[:,0], time_features[:,1], time_features[:,2], time_features[:,3], time_features[:,4], time_features[:,5]
        year_emb = self.year_embeddings(year)         # 10
        month_emb = self.month_embeddings(month)      # 5
        day_emb = self.day_embeddings(day)            # 5
        hour_emb = self.hour_embeddings(hour)         # 5
        weekday_emb = self.weekday_embeddings(weekday)# 5
        isweekend_emb = self.isweekend_embeddings(isweekend)  # 2

        time_emb = torch.cat([year_emb, month_emb, day_emb, hour_emb, weekday_emb, isweekend_emb], dim=1)  # 32

        interaction = torch.cat([combined_user * combined_book, combined_user, combined_book, time_emb], dim=1)  # 64 + 64 + 64 + 32 = 224
        interaction = self.dropout(interaction)
        output = self.output(interaction).squeeze()
        return output