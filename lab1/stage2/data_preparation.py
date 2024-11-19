# data_preparation.py

import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import ast
from transformers import BertTokenizer, BertModel

class BookRatingDataset(Dataset):
    def __init__(self, data, user_to_idx, book_to_idx, tag_embedding_dict):
        self.data = data
        self.user_to_idx = user_to_idx
        self.book_to_idx = book_to_idx
        self.tag_embedding_dict = tag_embedding_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        user = self.user_to_idx[row['User']]
        book = self.book_to_idx[row['Book']]
        rating = row['Rate'].astype('float32')
        text_embedding = self.tag_embedding_dict.get(row['Book'])
        return user, book, rating, text_embedding

def create_id_mapping(id_list):
    # 从ID列表中删除重复项并创建一个排序的列表
    unique_ids = sorted(set(id_list))
    
    # 创建将原始ID映射到连续索引的字典
    id_to_idx = {id: idx for idx, id in enumerate(unique_ids)}
    
    # 创建将连续索引映射回原始ID的字典
    idx_to_id = {idx: id for id, idx in id_to_idx.items()}
    
    return id_to_idx, idx_to_id

def load_tag_embeddings(loaded_data, tokenizer, model, device, save_path='data/tag_embedding_dict.pkl'):
    tag_embedding_dict = {}

    with torch.no_grad():
        for index, rows in tqdm(loaded_data.iterrows(), total=loaded_data.shape[0], desc='生成标签嵌入'):
            # 将标签列表转换为字符串
            tags_str = " ".join(rows.Tags)
            # 使用BERT中文模型对标签进行编码
            inputs = tokenizer(tags_str, truncation=True, return_tensors='pt').to(device)
            outputs = model(**inputs)
            # 使用最后一层的平均隐藏状态作为标签的向量表示
            tag_embedding = outputs.last_hidden_state.mean(dim=1).cpu()
            tag_embedding_dict[rows.Book] = tag_embedding

    # 将映射表存储为二进制文件
    with open(save_path, 'wb') as f:
        pickle.dump(tag_embedding_dict, f)
    
    return tag_embedding_dict

def load_existing_tag_embeddings(load_path='data/tag_embedding_dict.pkl'):
    with open(load_path, 'rb') as f:
        tag_embedding_dict = pickle.load(f)
    return tag_embedding_dict

def prepare_dataloaders(csv_path, tag_embedding_dict, test_size=0.5, batch_size=4096):
    loaded_data = pd.read_csv(csv_path)

    user_ids = loaded_data['User'].unique()
    book_ids = loaded_data['Book'].unique()

    user_to_idx, idx_to_user = create_id_mapping(user_ids)
    book_to_idx, idx_to_book = create_id_mapping(book_ids)

    # 划分训练集和测试集
    train_data, test_data = train_test_split(loaded_data, test_size=test_size, random_state=42)

    # 创建训练集和测试集的数据集对象
    train_dataset = BookRatingDataset(train_data, user_to_idx, book_to_idx, tag_embedding_dict)
    test_dataset = BookRatingDataset(test_data, user_to_idx, book_to_idx, tag_embedding_dict)

    # 创建训练集和测试集的数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, test_dataloader, len(user_ids), len(book_ids)