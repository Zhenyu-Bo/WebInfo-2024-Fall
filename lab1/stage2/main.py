# main.py

import torch
import pickle
import numpy as np
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import ast
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
import os

from data_preparation import (
    load_tag_embeddings,
    load_existing_tag_embeddings,
    prepare_dataloaders
)
from models import MatrixFactorization
from train import train_model

def main():

    hyperparams = {
        'use_tags': True,
        'embedding_dim': 32,
        'user_tag_embedding_dim': 100,  # 根据使用的embedding方法调整
        'book_tag_embedding_dim': 100,  # 根据使用的embedding方法调整
        'criterion': 'MSELoss',
        'optimizer': 'Adam',
        'learning_rate': 0.01,
        'num_epochs': 30,
        'lambda_u': 0.001,
        'lambda_b': 0.001,
        'lambda_time': 0.001,
        'lambda_output': 0.001,
        'dropout_rate': 0.5,
        'patience': 5,
        'batch_size': 4096,
        'test_size': 0.5,
        'embedding_method': 'word2vec'  # 可选 'tfidf', 'word2vec', 'bert'
    }

    # 根据嵌入方法动态设置标签嵌入维度
    if hyperparams['embedding_method'] == 'bert':
        hyperparams['user_tag_embedding_dim'] = 768
        hyperparams['book_tag_embedding_dim'] = 768
    else:
        hyperparams['user_tag_embedding_dim'] = 100
        hyperparams['book_tag_embedding_dim'] = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')

    # 加载用户数据
    user_data = pd.read_csv('../data/book_score.csv')
    user_data['Tag'] = user_data['Tag'].fillna('')
    user_data['Tag'] = user_data['Tag'].apply(lambda x: x.split(','))

    # 加载书籍数据
    book_data = pd.read_csv('../data/selected_book_top_1200_data_tag.csv')
    book_data['Tags'] = book_data['Tags'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

    if hyperparams['use_tags']:
        if hyperparams['embedding_method'] == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            model_bert = BertModel.from_pretrained('bert-base-chinese').to(device)
            # 加载用户标签嵌入
            try:
                user_tag_embedding_dict = load_existing_tag_embeddings('data/user_tag_embedding_dict_bert.pkl')
                print("加载已存在的用户BERT标签嵌入。")
            except FileNotFoundError:
                print("未找到用户标签嵌入文件，开始生成用户BERT标签嵌入。")
                user_tag_embedding_dict = load_tag_embeddings(user_data, id_column='User', tag_column='Tag', method='bert', 
                                                              tokenizer=tokenizer, model=model_bert, device=device, 
                                                              save_path='data/user_tag_embedding_dict_bert.pkl')
            # 加载书籍标签嵌入
            try:
                book_tag_embedding_dict = load_existing_tag_embeddings('data/book_tag_embedding_dict_bert.pkl')
                print("加载已存在的书籍BERT标签嵌入。")
            except FileNotFoundError:
                print("未找到书籍标签嵌入文件，开始生成书籍BERT标签嵌入。")
                book_tag_embedding_dict = load_tag_embeddings(book_data, id_column='Book', tag_column='Tags', method='bert', 
                                                              tokenizer=tokenizer, model=model_bert, device=device, 
                                                              save_path='data/book_tag_embedding_dict_bert.pkl')
        else:
            # 加载用户标签嵌入
            try:
                user_tag_embedding_dict = load_existing_tag_embeddings('data/user_tag_embedding_dict_' + hyperparams['embedding_method'] + '.pkl')
                print("加载已存在的用户标签嵌入。")
            except FileNotFoundError:
                print("未找到用户标签嵌入文件，开始生成用户标签嵌入。")
                user_tag_embedding_dict = load_tag_embeddings(user_data, id_column='User', tag_column='Tag', method=hyperparams['embedding_method'],
                                                              save_path='data/user_tag_embedding_dict_' + hyperparams['embedding_method'] + '.pkl')
            # 加载书籍标签嵌入
            try:
                book_tag_embedding_dict = load_existing_tag_embeddings('data/book_tag_embedding_dict_' + hyperparams['embedding_method'] + '.pkl')
                print("加载已存在的书籍标签嵌入。")
            except FileNotFoundError:
                print("未找到书籍标签嵌入文件，开始生成书籍标签嵌入。")
                book_tag_embedding_dict = load_tag_embeddings(book_data, id_column='Book', tag_column='Tags', method=hyperparams['embedding_method'],
                                                              save_path='data/book_tag_embedding_dict_' + hyperparams['embedding_method'] + '.pkl')
    else:
        user_tag_embedding_dict = None
        book_tag_embedding_dict = None
        print("不使用标签嵌入。")

    train_dataloader, test_dataloader, num_users, num_books = prepare_dataloaders(
        user_data_path='../data/book_score.csv',
        book_data_path='../data/selected_book_top_1200_data_tag.csv',
        user_tag_embedding_dict=user_tag_embedding_dict,
        book_tag_embedding_dict=book_tag_embedding_dict,
        test_size=hyperparams['test_size'],
        batch_size=hyperparams['batch_size'],
        user_tag_embedding_dim=hyperparams['user_tag_embedding_dim'],
        book_tag_embedding_dim=hyperparams['book_tag_embedding_dim']
    )

    model = MatrixFactorization(
        num_users, 
        num_books, 
        hyperparams['embedding_dim'], 
        hyperparams['user_tag_embedding_dim'],
        hyperparams['book_tag_embedding_dim'],
        hyperparams['use_tags'],
        dropout_rate=hyperparams['dropout_rate']
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join('model', f'training_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)

    num_epochs = hyperparams['num_epochs']
    lambda_u, lambda_b = hyperparams['lambda_u'], hyperparams['lambda_b']
    lambda_time = hyperparams['lambda_time']
    lambda_output = hyperparams['lambda_output']
    patience = hyperparams['patience']
    start_time = time.time()
    train_losses, test_losses, ndcg_scores_list = train_model(
        model, 
        train_dataloader, 
        test_dataloader, 
        criterion, 
        optimizer, 
        device, 
        num_epochs, 
        lambda_u, 
        lambda_b,
        lambda_time,
        lambda_output,
        patience
    )
    end_time = time.time()
    print(f"训练模型共耗时：{end_time - start_time:.2f}秒。")

    best_model_path = os.path.join(model_dir, f'best_matrix_factorization_{timestamp}.pth')
    torch.save(model.state_dict(), best_model_path)
    print(f"最佳模型已保存为 {best_model_path}。")

    hyperparams.update({
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'final_ndcg': ndcg_scores_list[-1]
    })
    config_filename = f'config_{timestamp}.json'
    with open(os.path.join(model_dir, config_filename), 'w') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)
    print(f"超参数已保存为 {config_filename}。")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='训练损失')
    plt.plot(epochs, test_losses, label='测试损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练和测试损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, ndcg_scores_list, label='平均NDCG')
    plt.xlabel('轮次')
    plt.ylabel('NDCG')
    plt.title('平均NDCG')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_curves.png'))
    plt.show()
    print(f"损失函数和NDCG曲线已保存为 {os.path.join(model_dir, 'training_curves.png')}。")

if __name__ == '__main__':
    main()