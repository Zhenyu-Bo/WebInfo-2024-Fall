import os
import sys
import random
from time import time

import pandas as pd
import matplotlib.pyplot as plt  # 新增导入
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model.Embedding_based import Embedding_based
from parsers.parser_Embedding_based import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_Embedding_based import DataLoader


def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode='predict')       # (n_batch_users, n_items)

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks, device)

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    # load data
    data = DataLoader(args, logging)

    # construct model & optimizer
    model = Embedding_based(args, data.n_users, data.n_items, data.n_entities, data.n_relations)
    if args.use_pretrain == 1:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}

    # 新增：用于存储每个 epoch 的损失
    cf_loss_list = []
    kg_loss_list = []

    # train model
    for epoch in range(1, args.n_epoch + 1):
        model.train()

        # 优化 CF 损失
        time1 = time()
        total_cf_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            optimizer.zero_grad()
            cf_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='cf')
            if torch.isnan(cf_loss):
                logging.info('ERROR: Epoch {:04d} Iter {:04d} / {:04d} CF Loss is nan.'.format(epoch, iter, n_cf_batch))
                sys.exit()
            cf_loss.backward()
            optimizer.step()
            total_cf_loss += cf_loss.item()

            if (iter % args.print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter CF Loss {:.4f} | CF Loss Mean {:.4f}'.format(
                    epoch, iter, n_cf_batch, cf_loss.item(), total_cf_loss / iter))

        avg_cf_loss = total_cf_loss / n_cf_batch
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total CF Loss {:.4f}'.format(
            epoch, n_cf_batch, avg_cf_loss))

        # 记录 CF 损失
        cf_loss_list.append(avg_cf_loss)

        # 优化 KG 损失
        total_kg_loss = 0
        n_kg_batch = data.n_kg_data // data.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.kg_dict, data.kg_batch_size, data.n_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            optimizer.zero_grad()
            kg_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode='kg')
            if torch.isnan(kg_loss):
                logging.info('ERROR: Epoch {:04d} Iter {:04d} / {:04d} KG Loss is nan.'.format(epoch, iter, n_kg_batch))
                sys.exit()
            kg_loss.backward()
            optimizer.step()
            total_kg_loss += kg_loss.item()

            if (iter % args.print_every) == 0:
                logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter KG Loss {:.4f} | KG Loss Mean {:.4f}'.format(
                    epoch, iter, n_kg_batch, kg_loss.item(), total_kg_loss / iter))

        avg_kg_loss = total_kg_loss / n_kg_batch
        logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total KG Loss {:.4f}'.format(
            epoch, n_kg_batch, avg_kg_loss))

        # 记录 KG 损失
        kg_loss_list.append(avg_kg_loss)

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time3 = time()
            _, metrics_dict = evaluate(model, data, Ks, device)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, time() - time3, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, log_save_id, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # 保存损失曲线图
    plt.figure()
    plt.plot(range(1, len(cf_loss_list) + 1), cf_loss_list, label='CF Loss')
    plt.plot(range(1, len(kg_loss_list) + 1), kg_loss_list, label='KG Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, 'training_loss.png'))
    plt.close()

    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics_Emb_iterative.tsv', sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info('Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)], best_metrics['ndcg@{}'.format(k_max)]))


def predict(args):
    # GPU / CPU
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    # load data
    data = DataLoader(args, logging)

    # load model
    model = Embedding_based(args, data.n_users, data.n_items, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    cf_scores, metrics_dict = evaluate(model, data, Ks, device)
    np.save(args.save_dir + '/cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))


if __name__ == '__main__':
    args = parse_args()
    train(args)
    # predict(args)