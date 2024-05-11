import numpy as np
import scipy.sparse as sp
import os.path as osp
import os
import urllib.request
import sys
import pickle as pkl
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


import zipfile
import json
import platform
from sklearn.model_selection import train_test_split

import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
import os.path as osp
from torch.nn.modules.module import Module

import torch.nn as nn
import math
from copy import deepcopy
from sklearn.metrics import f1_score
from scipy.io import loadmat
from scipy.sparse import csr_matrix
import scipy.io



from solverFM import *
from Datasets import Dataset, PrePtbDataset  
from defense import * 
from defense import GCN 
from utils import * 


def Parser():
    parser = argparse.ArgumentParser(description="RobustFM")
    parser.add_argument('--no_cuda', type=int, default=0, help="Whether to disable CUDA (0 for False, 1 for True)")
    parser.add_argument('--ptb_rate', type=float, default=0.1, help="Rate of perturbation for the PTB attack")
    parser.add_argument('--seed', type=int, default=15, help="Seed for random number generation")
    parser.add_argument('--dataset', type=str, default='citeseer', help="Name of the dataset to use, 'cora', 'citeseer'")
    parser.add_argument('--attack', type=str, default='meta', choices=['random','meta','nettack'],
                        help="Type of attack to perform")
    parser.add_argument('--hidden', type=int, default=16, help="Number of hidden units in the model")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate for regularization")
    parser.add_argument('--epochs', type=int, default=250, help="Number of training epochs")
    parser.add_argument('--only_gcn', action='store_true', help="Whether to use only GCN layer or full model")
    parser.add_argument('--read_clean', type=int, default=1, help="if you have the cleaned data saved, read it")


    return parser

def main(args):
    no_cuda = args.no_cuda
    ptb_rate = args.ptb_rate
    seed = args.seed
    dataset = args.dataset
    attack = args.attack   #random, meta, nettack
    hidden = args.hidden
    dropout = args.dropout
    epochs =  args.epochs
    only_gcn= only_gcn.args

    cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        torch.cuda.manual_seed(seed)
    if ptb_rate == 0:
        attack = "no"

    ############### DATA - import data ##############
    data = Dataset(root='/tmp/', name=dataset, setting='prognn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    if dataset == 'pubmed':
        print("just for matching the results in the paper," + "see details in https://github.com/ChandlerBang/Pro-GNN/issues/2")
        idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0],val_size=0.1, test_size=0.8, stratify=encode_onehot(labels), seed=15)

    if attack == 'no':
        perturbed_adj = adj

    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(np.array(features.todense()))

    ############### Perturb the data - Attach it either randomly, or using meta or nettach ##############
    if attack == 'random':
        from deeprobust.graph.global_attack import Random
        import random; random.seed(seed)
        np.random.seed(seed)
        attacker = Random()
        n_perturbations = int(ptb_rate * (adj.sum()//2))
        attacker.attack(adj, n_perturbations, type='add')
        perturbed_adj = attacker.modified_adj

    if attack == 'meta' or attack == 'nettack':
        perturbed_data = PrePtbDataset(root='/tmp/',name=dataset,attack_method=attack,ptb_rate=ptb_rate)
        perturbed_adj = perturbed_data.adj
        if attack == 'nettack':
            idx_test = perturbed_data.target_nodes

    np.random.seed(seed)
    torch.manual_seed(seed)


    ############### Set up the targeted model - a GCN ##############
    model = GCN(nfeat=features.shape[1],
            nhid=hidden,
            nclass=labels.max().item() + 1,
            dropout=dropout, device=device)

    ############## Save Noisy "Petreubed" data ############# 
    if args.save: 
        perturbed_adj
        dense_matrix = perturbed_adj.toarray()
        name_ = dataset+'_' + str(int(100*ptb_rate) )+ '_' + attack
        scipy.io.savemat('/content/drive/MyDrive/Colab_Notebooks/Proj_10_RobustMF/'+name_+'.mat', {name_: list(dense_matrix)})

    if args.read_clean:           #### read the cleaned saved data
        name_2 = 'clean_'+ name_ + '.mat'
        M =10 
        data = loadmat('/content/drive/MyDrive/Colab_Notebooks/Proj_10_RobustMF/'+name_2)
        perturbed_adj0 = data['M']
        adj_clean = torch.FloatTensor(perturbed_adj0)

        ### Binirize 
        thr = 0.1
        adj_clean[adj_clean >= thr] = 1
        model.fit(features, adj_clean, labels, idx_train, idx_val, verbose=False, train_iters=epochs)
        model.test(idx_test)
    else:           ######## Run the defense MF stratigy 
        perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, device=device)
        adj_new = symnmf_newton(perturbed_adj, k=10)   # k is the latent feature vector
        model.fit(features, adj_new, labels, idx_train, idx_val, verbose=False, train_iters=args.epochs)
        model.test(idx_test)
        
   
if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()
    main(args)

