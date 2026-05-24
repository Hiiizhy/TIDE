import torch
import numpy as np
import math
from utils import generateBatchSamples

def evaluating(model, dataLoader, config, device, isTrain, global_graph_data):

    evalBatchSize = config.batchSize
    numUser = dataLoader.numValid if isTrain else dataLoader.numTest
    idxList = list(range(numUser))
    numBatch = (numUser + evalBatchSize - 1) // evalBatchSize

    model.eval()
    predIdxArray = None
    targetList = []

    for batch in range(numBatch):
        start = batch * evalBatchSize
        end = min(start + evalBatchSize, numUser)
        batchList = idxList[start:end]

        (x_np, o_decay, uhis, target, f_seq, d_t) = generateBatchSamples(dataLoader, batchList, config, isEval=1)

        x = torch.from_numpy(x_np).long().to(device)
        o_decay = torch.from_numpy(o_decay).float().to(device)
        uhis = torch.from_numpy(uhis).float().to(device)
        d_t, f_seq = d_t.to(device), f_seq.to(device)

        with torch.no_grad():
            scores, _, _, _ = model(x, o_decay, d_t, global_graph_data, f_seq, uhis, isEval=True)

        scores[:, 0] = -1e9
        _, predIdx = torch.topk(scores, 100, largest=True)
        predIdx = predIdx.cpu().numpy()

        predIdxArray = predIdx if predIdxArray is None else np.append(predIdxArray, predIdx, axis=0)
        targetList += target

    Recall, NDCG = [], []
    for k in [10, 20]:
        Recall.append(calRecall(targetList, predIdxArray, k))
        NDCG.append(calNDCG(targetList, predIdxArray, k))

    return Recall, NDCG

def calRecall(target, pred, k):
    sumRecall = 0
    for i in range(len(target)):
        gt = set(target[i])
        ptar = set(pred[i][:k])
        if gt:
            sumRecall += len(gt & ptar) / float(len(gt))
    return sumRecall / float(len(target))

def calNDCG(target, pred, k):
    sumNDCG = 0
    for i in range(len(target)):
        gt = set(target[i])
        if not gt: continue
        
        dcg = sum([int(pred[i][j] in gt) / math.log(j + 2, 2) for j in range(k)])
        idcg = sum([1.0 / math.log(j + 2, 2) for j in range(min(k, len(gt)))])
        
        sumNDCG += dcg / idcg if idcg > 0 else 0
    return sumNDCG / float(len(target))