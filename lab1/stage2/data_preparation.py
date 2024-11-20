# data_preparation.py

import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import ast
import os

class BookRatingDataset(Dataset):
    def __init__(self, data, user_to_idx, book_to_idx, user_tag_embedding_dict, book_tag_embedding_dict, 
                 user_tag_embedding_dim=768, book_tag_embedding_dim=768):
        self.data = data
        self.user_to_idx = user_to_idx
        self.book_to_idx = book_to_idx
        self.user_tag_embedding_dict = user_tag_embedding_dict
        self.book_tag_embedding_dict = book_tag_embedding_dict
        self.user_tag_embedding_dim = user_tag_embedding_dim
        self.book_tag_embedding_dim = book_tag_embedding_dim

        # 处理时间特征
        self.data['Timestamp'] = pd.to_datetime(self.data['Time'])
        self.data['Year'] = self.data['Timestamp'].dt.year
        self.data['Month'] = self.data['Timestamp'].dt.month
        self.data['Day'] = self.data['Timestamp'].dt.day
        self.data['Hour'] = self.data['Timestamp'].dt.hour
        self.data['Weekday'] = self.data['Timestamp'].dt.weekday
        self.data['IsWeekend'] = self.data['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        user = self.user_to_idx[row['User']]
        book = self.book_to_idx[row['Book']]
        rating = row['Rate'].astype('float32')
        
        if self.user_tag_embedding_dict is not None:
            user_tag_embedding = self.user_tag_embedding_dict.get(row['User'], torch.zeros(self.user_tag_embedding_dim))
        else:
            user_tag_embedding = torch.zeros(self.user_tag_embedding_dim)
        
        if self.book_tag_embedding_dict is not None:
            book_tag_embedding = self.book_tag_embedding_dict.get(row['Book'], torch.zeros(self.book_tag_embedding_dim))
        else:
            book_tag_embedding = torch.zeros(self.book_tag_embedding_dim)

        # 获取时间特征
        time_features = torch.tensor([
            row['Year'] - 2000,
            row['Month'] - 1,
            row['Day']- 1,
            row['Hour'],
            row['Weekday'],
            row['IsWeekend']
        ], dtype=torch.long)

        return user, book, rating, user_tag_embedding, book_tag_embedding, time_features

def create_id_mapping(id_list):
    unique_ids = sorted(set(id_list))
    id_to_idx = {id: idx for idx, id in enumerate(unique_ids)}
    return id_to_idx, {idx: id for id, idx in id_to_idx.items()}

def load_tag_embeddings(loaded_data, id_column, tag_column, method='tfidf', tokenizer=None, model=None, device=None, save_path=None):
    tag_embedding_dict = {}
    ids = loaded_data[id_column].unique()
    id_to_tags = {}  # 用于存储每个ID对应的所有标签

    for id in ids:
        tags = loaded_data[loaded_data[id_column]==id][tag_column]
        tags = tags.dropna().tolist()
        tags_flat = []
        for tag_list in tags:
            if isinstance(tag_list, list):
                tags_flat.extend(tag_list)
            else:
                tags_flat.extend(str(tag_list).split(','))
        id_to_tags[id] = tags_flat

    if method == 'tfidf':
        all_tags = [' '.join(tags) for tags in id_to_tags.values()]
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(all_tags)
        for idx, id in enumerate(ids):
            tag_embedding = torch.tensor(tfidf_matrix[idx].toarray()[0], dtype=torch.float)
            tag_embedding_dict[id] = tag_embedding
    elif method == 'word2vec':
        sentences = list(id_to_tags.values())
        model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
        for id in ids:
            vectors = [torch.tensor(model_w2v.wv[word], dtype=torch.float) for word in id_to_tags[id] if word in model_w2v.wv]
            if vectors:
                tag_embedding = torch.mean(torch.stack(vectors), dim=0)
            else:
                tag_embedding = torch.zeros(100)
            tag_embedding_dict[id] = tag_embedding
    elif method == 'bert':
        with torch.no_grad():
            for id in tqdm(ids, desc=f'生成{tag_column}嵌入'):
                tags_str = " ".join(id_to_tags[id])
                inputs = tokenizer(tags_str, truncation=True, return_tensors='pt').to(device)
                outputs = model(**inputs)
                tag_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
                tag_embedding_dict[id] = tag_embedding
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(tag_embedding_dict, f)
    return tag_embedding_dict

def load_existing_tag_embeddings(load_path):
    with open(load_path, 'rb') as f:
        return pickle.load(f)

def prepare_dataloaders(user_data_path, book_data_path, user_tag_embedding_dict, book_tag_embedding_dict, test_size=0.5, batch_size=4096, 
                        user_tag_embedding_dim=768, book_tag_embedding_dim=768):
    loaded_data = pd.read_csv(user_data_path)
    loaded_data['Tag'] = loaded_data['Tag'].fillna('')
    loaded_data['Tag'] = loaded_data['Tag'].apply(lambda x: x.split(','))

    user_to_idx, idx_to_user = create_id_mapping(loaded_data['User'])
    book_to_idx, idx_to_book = create_id_mapping(loaded_data['Book'])

    train_data, test_data = train_test_split(loaded_data, test_size=test_size, random_state=42)

    train_dataset = BookRatingDataset(train_data, user_to_idx, book_to_idx, user_tag_embedding_dict, book_tag_embedding_dict, 
                                      user_tag_embedding_dim, book_tag_embedding_dim)
    test_dataset = BookRatingDataset(test_data, user_to_idx, book_to_idx, user_tag_embedding_dict, book_tag_embedding_dict, 
                                     user_tag_embedding_dim, book_tag_embedding_dim)

    def collate_fn(batch):
        users, books, ratings, user_tag_embeddings, book_tag_embeddings, time_features = zip(*batch)
        users = torch.tensor(users, dtype=torch.long)
        books = torch.tensor(books, dtype=torch.long)
        ratings = torch.tensor(ratings, dtype=torch.float32)
        user_tag_embeddings = torch.stack(user_tag_embeddings)
        book_tag_embeddings = torch.stack(book_tag_embeddings)
        time_features = torch.stack(time_features)
        return users, books, ratings, user_tag_embeddings, book_tag_embeddings, time_features

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)

    return train_dataloader, test_dataloader, len(user_to_idx), len(book_to_idx)