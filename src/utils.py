import numpy as np
import torch
import torch.nn.functional as F
import math
from collections import defaultdict

def generateBatchSamples(dataLoader, batchIdx, config, isEval):    
    samples, sampleLen, userhis, _, _, target = dataLoader.batchLoader(batchIdx, config.isTrain, isEval)
    
    maxLenSeq = max([len(userLen) for userLen in sampleLen])
    maxLenBas = max([max(userLen) for userLen in sampleLen])

    paddedSamples = []
    paddedDecays  = []
    paddedFreq    = []
    paddedDeltaT  = []

    for user in samples:
        trainU = user[:-1] 
        
        paddedU, decayU, freqU, deltaTU = [], [], [], []
        
        item_counter = defaultdict(int)
        seq_len = len(trainU)

        for t, eachBas in enumerate(trainU):
            padded_bas_ids = eachBas + [config.padIdx] * (maxLenBas - len(eachBas))
            paddedU.append(padded_bas_ids)
            
            decayU.append(config.decay ** (seq_len - 1 - t))
            
            bas_freq = []
            for item_id in eachBas:
                item_counter[item_id] += 1
                bas_freq.append(item_counter[item_id])
            padded_bas_freq = bas_freq + [0] * (maxLenBas - len(eachBas))
            freqU.append(padded_bas_freq)
            
            dist = float(seq_len - t)
            padded_bas_dist = [dist] * len(eachBas) + [0.0] * (maxLenBas - len(eachBas))
            deltaTU.append(padded_bas_dist)

        padding_len = maxLenSeq - len(paddedU)
        if padding_len > 0:
            paddedU  += [[config.padIdx] * maxLenBas] * padding_len
            decayU   += [0] * padding_len
            freqU    += [[0] * maxLenBas] * padding_len
            deltaTU  += [[0.0] * maxLenBas] * padding_len

        paddedSamples.append(paddedU)
        paddedDecays.append(decayU)
        paddedFreq.append(freqU)
        paddedDeltaT.append(deltaTU)

    return (
        np.asarray(paddedSamples),
        np.asarray(paddedDecays).reshape(len(samples), -1, 1), 
        userhis, 
        target,  
        torch.LongTensor(paddedFreq),  
        torch.FloatTensor(paddedDeltaT).unsqueeze(-1)  
    )