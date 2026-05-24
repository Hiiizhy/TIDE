import argparse
import os
import torch
from trainer import training
from data_loader import dataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TIDE: Disentangling Habit and Exploration")
    
    parser.add_argument('--dataset', type=str, default='beauty')
    parser.add_argument('--device', type=str, default='cuda', help='cpu, cuda, cuda:0, cuda:1')
    parser.add_argument('--mode', type=str, default='MODEL')
    parser.add_argument('--word', type=str, default='main', help='Experiment suffix')

    parser.add_argument('--batchSize', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--numIter', type=int, default=300)

    parser.add_argument('--cl_weight', type=float, default=0.5, help='Lambda (λ): Contrastive loss weight')
    parser.add_argument('--support_count', type=int, default=5, help='s: Minimum support for pattern graph')
    parser.add_argument('--decay', type=float, default=0.1, help='Beta (β): Hawkes temporal decay rate')
    parser.add_argument('--isTrain', type=int, default=1)
    parser.add_argument('--evalEpoch', type=int, default=5)
    parser.add_argument('--testOrder', type=int, default=1)

    config = parser.parse_args()

    if config.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"Warning: CUDA not available. Falling back to CPU.")
        config.device = torch.device('cpu')
    else:
        config.device = torch.device(config.device)
    
    print(f">>> Active Device: {config.device}")

    res_dir = os.path.join('results', config.dataset)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    dataset = dataLoader(config.dataset, config)
    config.padIdx = dataset.numItemsTrain if config.isTrain else dataset.numItemsTest

    print(f"--- Starting Session: {config.dataset} | Mode: {config.mode} ---")
    training(dataset, config, config.device)
    print("--- Session Ended ---")