# predict.py untested

import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import ast
import os
import json

from data_preparation import load_existing_tag_embeddings
from models import MatrixFactorization

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')

    # 指定模型目录和文件名（需要替换为您的实际路径）
    model_dir = 'model'  # 模型保存的目录
    # 获取最新的模型目录
    model_subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    latest_model_subdir = sorted(model_subdirs)[-1]
    model_path = os.path.join(model_dir, latest_model_subdir)

    # 加载超参数配置
    config_filename = [f for f in os.listdir(model_path) if f.startswith('config_')][0]
    with open(os.path.join(model_path, config_filename), 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    # 加载模型参数
    model_filename = [f for f in os.listdir(model_path) if f.startswith('matrix_factorization_')][0]
    model_state_dict = torch.load(os.path.join(model_path, model_filename))

    # 初始化模型
    embedding_dim = hyperparams['embedding_dim']
    hidden_state = hyperparams['hidden_state']
    num_users = model_state_dict['user_embeddings.weight'].shape[0]
    num_books = model_state_dict['book_embeddings.weight'].shape[0]

    model = MatrixFactorization(num_users, num_books, embedding_dim, hidden_state).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    # 加载标签嵌入字典
    tag_embedding_dict = load_existing_tag_embeddings('data/tag_embedding_dict.pkl')

    # 加载物品数据
    item_data = pd.read_csv('../data/selected_book_top_1200_data_tag.csv')
    item_data['Tags'] = item_data['Tags'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

    # 准备物品ID映射（确保与训练时的编码一致）
    item2id = {item_id: idx for idx, item_id in enumerate(item_data['BookID'].unique())}
    id2item = {idx: item_id for item_id, idx in item2id.items()}

    # 准备标签嵌入
    tag_embeddings = []
    for tags in item_data['Tags']:
        embeddings = [tag_embedding_dict[tag] for tag in tags if tag in tag_embedding_dict]
        if embeddings:
            tag_embedding = np.mean(embeddings, axis=0)
        else:
            tag_embedding = np.zeros((hidden_state,))
        tag_embeddings.append(tag_embedding)
    tag_embeddings = torch.tensor(tag_embeddings, dtype=torch.float).to(device)

    # 指定要进行预测的用户ID（需要替换为实际的用户ID，确保与训练时的编码一致）
    user_id = 0  # 示例用户ID
    user_ids = torch.tensor([user_id] * num_books, dtype=torch.long).to(device)

    # 准备物品ID张量
    item_ids = torch.tensor(list(id2item.keys()), dtype=torch.long).to(device)

    # 进行预测
    with torch.no_grad():
        predictions = model(user_ids, item_ids, tag_embeddings)

    # 获取预测评分并添加到数据中
    predictions = predictions.cpu().numpy()
    item_data['PredictedRating'] = predictions

    # 按预测评分排序，获取Top-N推荐
    top_N = 10
    recommendations = item_data.sort_values(by='PredictedRating', ascending=False).head(top_N)

    # 输出推荐结果
    print(f"为用户 {user_id} 推荐的Top {top_N}本书籍：")
    print(recommendations[['BookID', 'BookTitle', 'PredictedRating']])
    
if __name__ == '__main__':
    main()