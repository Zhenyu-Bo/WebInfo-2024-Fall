# train.py

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils import compute_ndcg
from sklearn.metrics import ndcg_score
import os

def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, device, num_epochs=20, 
               lambda_u=0.001, lambda_b=0.001, lambda_time=0.001, lambda_output=0.001, 
               patience=5):
    train_losses = []
    test_losses = []
    ndcg_scores_list = []
    best_ndcg = -np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss_train = 0.0

        for idx, (user_ids, book_ids, ratings, tag_embedding, time_features) in tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            desc=f'Epoch {epoch+1}'
        ):
            optimizer.zero_grad()
            
            predictions = model(
                user_ids.to(device), 
                book_ids.to(device), 
                tag_embedding.to(device),
                time_features.to(device)
            )
            
            loss = criterion(predictions, ratings.to(device)) + \
                lambda_u * model.user_embeddings.weight.norm(2) + \
                lambda_b * model.book_embeddings.weight.norm(2) + \
                lambda_time * model.year_embeddings.weight.norm(2) + \
                lambda_time * model.month_embeddings.weight.norm(2) + \
                lambda_time * model.day_embeddings.weight.norm(2) + \
                lambda_time * model.hour_embeddings.weight.norm(2) + \
                lambda_time * model.weekday_embeddings.weight.norm(2) + \
                lambda_time * model.isweekend_embeddings.weight.norm(2) + \
                lambda_output * model.output.weight.norm(2)
            
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()

        output_loss_train = total_loss_train / (idx + 1)
        train_losses.append(output_loss_train)
        print(f'第{epoch+1}轮，平均训练损失：{output_loss_train}')

        # 评估阶段
        model.eval()
        total_loss_test = 0.0
        results = []

        with torch.no_grad():
            for idx, (user_ids, item_ids, true_ratings, tag_embeddings, time_features) in tqdm(
                enumerate(test_dataloader), 
                total=len(test_dataloader), 
                desc=f'Evaluating Epoch {epoch+1}'
            ):
                pred_ratings = model(
                    user_ids.to(device), 
                    item_ids.to(device), 
                    tag_embeddings.to(device),
                    time_features.to(device)
                )

                loss = criterion(pred_ratings, true_ratings.to(device))
                total_loss_test += loss.item()

                # 收集结果用于计算NDCG
                user_ids_np = user_ids.cpu().numpy()
                pred_ratings_np = pred_ratings.cpu().numpy()
                true_ratings_np = true_ratings.cpu().numpy()

                # 将这三个 arrays 合并成一个 2D array
                batch_results = np.column_stack((user_ids_np, pred_ratings_np, true_ratings_np))

                # 将这个 2D array 添加到 results
                results.append(batch_results)

        # 将结果的 list 转换为一个大的 numpy array
        results = np.vstack(results)

        # 将结果转换为DataFrame
        results_df = pd.DataFrame(results, columns=['user', 'pred', 'true'])
        results_df['user'] = results_df['user'].astype(int)

        ndcg_scores = results_df.groupby('user').apply(compute_ndcg)

        # 计算平均NDCG
        avg_ndcg = ndcg_scores.mean()
        test_losses.append(total_loss_test / (idx + 1))
        ndcg_scores_list.append(avg_ndcg)
        print(f'第{epoch+1}轮，训练损失：{output_loss_train}, 测试损失：{total_loss_test / (idx + 1)}, 平均NDCG: {avg_ndcg}')

        # 检查早停条件
        if avg_ndcg > best_ndcg:
            best_ndcg = avg_ndcg
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'早停触发，训练在第{epoch+1}轮停止。')
                # 加载最佳模型
                model.load_state_dict(torch.load('best_model.pth'))
                break

    return train_losses, test_losses, ndcg_scores_list