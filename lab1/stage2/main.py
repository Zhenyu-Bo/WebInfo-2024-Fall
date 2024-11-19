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

from data_preparation import (
    load_tag_embeddings,
    load_existing_tag_embeddings,
    prepare_dataloaders
)
from models import MatrixFactorization
from train import train_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')

    # 初始化BERT模型和分词器
    print("正在加载BERT模型和分词器……")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model_bert = BertModel.from_pretrained('bert-base-chinese').to(device)

    # 读取CSV文件
    print("正在加载数据……")
    loaded_data = pd.read_csv('..\data\selected_book_top_1200_data_tag.csv')
    loaded_data['Tags'] = loaded_data['Tags'].apply(lambda x: ast.literal_eval(x))

    # 检查是否已有标签嵌入
    try:
        tag_embedding_dict = load_existing_tag_embeddings('data/tag_embedding_dict.pkl')
        print("加载已存在的标签嵌入。")
    except FileNotFoundError:
        print("未找到标签嵌入文件，开始生成标签嵌入。")
        tag_embedding_dict = load_tag_embeddings(loaded_data, tokenizer, model_bert, device)

    # 准备数据加载器
    train_dataloader, test_dataloader, num_users, num_books = prepare_dataloaders(
        csv_path='../data/book_score.csv',
        tag_embedding_dict=tag_embedding_dict,
        test_size=0.5,
        batch_size=4096
    )

    # 定义模型
    embedding_dim, hidden_state = 32, 768
    model = MatrixFactorization(num_users, num_books, embedding_dim, hidden_state).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    num_epochs = 20
    lambda_u, lambda_b = 0.001, 0.001
    start_time = time.time()
    train_losses, test_losses, ndcg_scores_list = train_model(model, train_dataloader, test_dataloader, criterion, optimizer, device, num_epochs, lambda_u, lambda_b)
    end_time = time.time()
    print(f"训练模型共耗时：{end_time - start_time:.2f}秒。")
    
    # 保存模型
    import os
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), './model/matrix_factorization.pth')
    print("模型已保存。")

    # 设置中文字体和解决负号显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False     # 解决坐标轴负号显示问题

    # 绘制损失函数曲线
    epochs = range(1, num_epochs + 1)
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
    plt.show()

if __name__ == '__main__':
    main()