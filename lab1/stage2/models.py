# models.py

import torch
from torch import nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim, hidden_state, use_tags=True, dropout_rate=0.5):
        super(MatrixFactorization, self).__init__()
        self.use_tags = use_tags
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.book_embeddings = nn.Embedding(num_books, embedding_dim)

        # 手动设置每个时间特征的嵌入维度
        self.year_embedding_dim = 10  # 分配更多的维度给年份
        self.month_embedding_dim = 5
        self.day_embedding_dim = 5
        self.hour_embedding_dim = 5
        self.weekday_embedding_dim = 5
        self.isweekend_embedding_dim = 2

        # 确保总的时间嵌入维度等于 embedding_dim
        total_time_embedding_dim = (self.year_embedding_dim + self.month_embedding_dim +
                                    self.day_embedding_dim + self.hour_embedding_dim +
                                    self.weekday_embedding_dim + self.isweekend_embedding_dim)
        assert total_time_embedding_dim == embedding_dim, "时间嵌入维度之和必须等于 embedding_dim"

        # 定义时间特征的嵌入层
        self.year_embeddings = nn.Embedding(20, self.year_embedding_dim)    # 年份跨度不超过20年
        self.month_embeddings = nn.Embedding(13, self.month_embedding_dim)  # 月份从1到12
        self.day_embeddings = nn.Embedding(32, self.day_embedding_dim)      # 日期从1到31
        self.hour_embeddings = nn.Embedding(24, self.hour_embedding_dim)    # 小时从0到23
        self.weekday_embeddings = nn.Embedding(7, self.weekday_embedding_dim)   # 星期从0到6
        self.isweekend_embeddings = nn.Embedding(2, self.isweekend_embedding_dim)   # 是否周末

        if self.use_tags:
            self.linear_embedding = nn.Linear(hidden_state, embedding_dim)
        
        self.dropout = nn.Dropout(p=dropout_rate)  # 引入Dropout层
        self.output = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user, book, tag_embedding, time_features):
        user_embedding = self.user_embeddings(user)
        book_embedding = self.book_embeddings(book)

        if self.use_tags:
            tag_embedding_proj = self.linear_embedding(tag_embedding)
            book_integrate = book_embedding + tag_embedding_proj
        else:
            book_integrate = book_embedding

        # 处理时间特征
        year = time_features[:, 0]
        month = time_features[:, 1]
        day = time_features[:, 2]
        hour = time_features[:, 3]
        weekday = time_features[:, 4]
        isweekend = time_features[:, 5]

        year_embedding = self.year_embeddings(year)
        month_embedding = self.month_embeddings(month)
        day_embedding = self.day_embeddings(day)
        hour_embedding = self.hour_embeddings(hour)
        weekday_embedding = self.weekday_embeddings(weekday)
        isweekend_embedding = self.isweekend_embeddings(isweekend)

        # 汇总时间嵌入
        time_embedding = torch.cat([
            year_embedding, 
            month_embedding, 
            day_embedding, 
            hour_embedding, 
            weekday_embedding, 
            isweekend_embedding
        ], dim=1)

        interaction = torch.cat([user_embedding * book_integrate, time_embedding], dim=1)
        interaction = self.dropout(interaction)  # 应用Dropout

        output = self.output(interaction).squeeze()
        return output