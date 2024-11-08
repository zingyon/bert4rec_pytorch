import numpy as np
import torch

def compute_metric(pred):
    logits, labels = pred
    # print(logits.shape, labels.shape)
    results = {}
    num_samples, vocab_size = logits.shape
    # 마스크 생성: labels이 -100이 아닌 샘플만 선택
    mask = labels != -100
    filtered_logits = logits
    filtered_labels = labels[mask]
    # print(filtered_logits.shape, filtered_labels.shape, labels.shape)
    # Correct logits 선택

    filtered_logits = torch.tensor(filtered_logits)
    filtered_labels = torch.tensor(filtered_labels)
    correct_logits = filtered_logits[torch.arange(num_samples), filtered_labels]
    
    """ 원본
    # 랭크 계산: correct_logits보다 큰 logits의 수 + 1
    ranks = (filtered_logits > correct_logits.unsqueeze(-1)).sum(dim=-1) + 1
    
    # DCG 계산 , 로그는 0에 대해 정의되지 않기 때문에, 순위가 0일 때 계산을 방지하기 위해 1을 더함
    dcg = 1.0 / torch.log2(ranks.float() + 1.0)
    
    for k in [1, 5, 10, 20]:
        # Hit@k 계산
        hit_k = (ranks <= k).float().mean().item()
        results[f"H@{k}"] = hit_k
        
        # NDCG@k 계산
        dcg_k = dcg * (ranks <= k).float()
        # k 이내에 있는 샘플 수로 나눔 (0으로 나누는 것을 방지하기 위해 clamp)
        ndcg_k = dcg_k.sum() / (ranks <= k).float().sum().item()
        results[f"NDCG@{k}"] = ndcg_k
    
    return results
    """


    # 랭크 계산: correct_logits보다 큰 logits의 수 + 1
    ranks = (filtered_logits > correct_logits.unsqueeze(-1)).sum(dim=-1) + 1

    # DCG 계산: 로그는 0에 대해 정의되지 않기 때문에, 순위가 0일 때 계산을 방지하기 위해 1을 더함
    dcg = 1.0 / torch.log2(ranks.float() + 1.0)

    for k in [5, 10, 15, 20, 40]:
        # Hit@k 계산
        hit_k = (ranks <= k).float().mean().item()
        results[f"H@{k}"] = hit_k

        # NDCG@k 계산
        dcg_k = dcg * (ranks <= k).float()  # DCG@k 계산
#        ndcg_k = dcg_k.sum() / (ranks <= k).float().sum().item()  # k 이내에 있는 샘플 수로 나눔

        # NDCG@k는 DCG@k / IDCG@k로 계산
        ndcg_k = dcg_k

        results[f"NDCG@{k}"] = ndcg_k.mean()

    return results
