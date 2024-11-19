# utils.py

from sklearn.metrics import ndcg_score

def compute_ndcg(group, k=50):
    true_ratings = group['true'].tolist()
    pred_ratings = group['pred'].tolist()
    return ndcg_score([true_ratings], [pred_ratings], k=k)