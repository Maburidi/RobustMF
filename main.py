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


from solverFM import *
from Datasets import Dataset, PrePtbDataset  
from defense import * 
from defense import GCN 



#from deeprobust.graph.defense import GCN
#from deeprobust.graph.data import Dataset, PrePtbDataset
#from deeprobust.graph.utils import preprocess, encode_onehot, get_train_val_test

def Parser():
    parser = argparse.ArgumentParser(description="Argument parser for custom options")
    parser.add_argument('--no_cuda', type=int, default=0, help="Whether to disable CUDA (0 for False, 1 for True)")
    parser.add_argument('--ptb_rate', type=float, default=0.1, help="Rate of perturbation for the PTB attack")
    parser.add_argument('--seed', type=int, default=15, help="Seed for random number generation")
    parser.add_argument('--dataset', type=str, default='citeseer', help="Name of the dataset to use")
    parser.add_argument('--attack', type=str, default='meta', choices=['no', 'random', 'meta', 'nettack'],
                        help="Type of attack to perform")
    parser.add_argument('--hidden', type=int, default=16, help="Number of hidden units in the model")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate for regularization")
    parser.add_argument('--epochs', type=int, default=250, help="Number of training epochs")
    parser.add_argument('--only_gcn', action='store_true', help="Whether to use only GCN layer or full model")

    return parser

def main(args):
   


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()
    main(args)





