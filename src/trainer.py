import torch
import numpy as np
import time
import os
import itertools
from collections import defaultdict
import torch.nn.functional as F
from model import Model, ContrastiveLoss, GlobalGraphBuilder
from utils import generateBatchSamples
from evaluation import evaluating

def training(dataLoader, config, device):
    is_train = config.isTrain
    numUsers = dataLoader.numTrain if is_train else dataLoader.numTrainVal
    numItems = dataLoader.numItemsTrain if is_train else dataLoader.numItemsTest
    raw_source = getattr(dataLoader, 'trainList' if is_train else 'trainValList', [])

    numBatch = (numUsers + config.batchSize - 1) // config.batchSize
    idxList = list(range(numUsers))

    print(">>> [Init] Building Global Item-Pattern Graph...")
    all_baskets = [b for seq in raw_source for b in seq if len(b) >= 2]
    
    pair_counts = defaultdict(int)
    for basket in all_baskets:
        unique_items = sorted(list(set(basket)))[:20]
        for pair in itertools.combinations(unique_items, 2):
            pair_counts[pair] += 1
            
    real_patterns = [list(p) for p, c in pair_counts.items() if c >= config.support_count]
    if not real_patterns: real_patterns = [[1, 2]]

    gb = GlobalGraphBuilder(numItems, min_support=config.support_count)
    graph_data = gb.build_graph_tensor(real_patterns, device=device)
    config.num_patterns = graph_data['num_patterns']

    model = Model(config, numItems).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    cl_criterion = ContrastiveLoss(temperature=0.1)

    for epoch in range(config.numIter):
        model.train()
        epochLoss = 0
        t_start = time.time()

        for batch in range(numBatch):
            start_idx = config.batchSize * batch
            end_idx = min(numUsers, start_idx + config.batchSize)
            
            x_np, o_decay, uhis, target_np, f_seq, d_t = generateBatchSamples(dataLoader, idxList[start_idx:end_idx], config, isEval=0)

            x = torch.from_numpy(x_np).long().to(device)
            o_decay, uhis = torch.from_numpy(o_decay).float().to(device), torch.from_numpy(uhis).float().to(device)
            target = torch.from_numpy(target_np).float().to(device)
            f_seq, d_t = f_seq.to(device), d_t.to(device)

            scores, h_seq, h_expl, _ = model(x, o_decay, d_t, graph_data, f_seq, uhis, isEval=False)

            if scores.shape[-1] > target.shape[-1]:
                target = F.pad(target, (0, scores.shape[-1] - target.shape[-1]), "constant", 0)

            loss = Bernoulli(target, scores) + config.cl_weight * cl_criterion(h_seq, h_expl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()

        print(f"Epoch {epoch} | Loss: {epochLoss/numBatch:.4f} | Time: {time.time()-t_start:.1f}s")

        if (epoch + 1) % config.evalEpoch == 0:
            recall, ndcg = evaluating(model, dataLoader, config, device, is_train, graph_data)
            
            r10, r20 = recall[0], recall[1]
            n10, n20 = ndcg[0], ndcg[1]
            
            print(f">>> Eval | R@10: {r10:.4f}, N@10: {n10:.4f} | R@20: {r20:.4f}, N@20: {n20:.4f}")

def Bernoulli(ground_truth, scores):
    scores = torch.clamp(scores, min=1e-10, max=1.0)
    return -(torch.log(scores) * ground_truth).sum(-1).mean()