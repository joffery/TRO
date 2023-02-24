import torch
import numpy as np
import random
import pickle
import argparse
import os

from torch.utils.data import DataLoader
from data_loader.data_loader import ToyDataset, SeqToyDataset

import torch.backends.cudnn as cudnn

from model.model import ERM as Model

from utils.centrality import get_centrality
from utils.utils import *
from utils.graph_utils import *

parser = argparse.ArgumentParser(description='Topology-Aware Distributionally Robust Optimization')
parser.add_argument('--dataset', default='toy_d15', type=str, help='toy_d15, toy_d60')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run ERM')
parser.add_argument('--batch_size', default=10, type=int, help='mini-batch size (default: 10)')
parser.add_argument('--lr_e',  default=3e-5, type=float, help='initial learning rate')
parser.add_argument('--gpu_id', default=1, type=int, help='gpu id')

# Use diffusion EMD or mean/var
parser.add_argument('--diffusion', default=1, type=int, help='use diffusion EMD or mean/var')

cudnn.benchmark = True

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 on stackoverflow
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

if args.dataset == "toy_d15":
    from configs.dg15 import opt
elif args.dataset == "toy_d60":
    from configs.dg60 import opt
else:
    raise NotImplementedError("Dataset not implemented. Please try toy_d15 or toy_d60!")

np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

opt.model = "ERM"
opt.dataset = args.dataset
opt.diffusion = args.diffusion
opt.outf = None

print("dataset: {}".format(opt.dataset))

# Hyper-params
opt.batch_size = args.batch_size
opt.lr_e = args.lr_e

# Dataset
if args.dataset == "toy_d15":
    opt.num_domain = 15
    # the specific source and target domain:
    opt.src_domain = [0, 12, 3, 4, 14, 8]
elif args.dataset == "toy_d60":
    opt.num_domain = 60
    # the specific source and target domain:
    opt.src_domain = list(range(6))
else:
    raise NotImplementedError("Dataset not implemented. Please try toy_d15 or toy_d60!")

opt.src_domain = np.array(opt.src_domain)
opt.num_source = opt.src_domain.shape[0]
opt.num_target = opt.num_domain - opt.num_source

if args.dataset == "toy_d15":
    data_source = os.path.join("data", "toy_d15_spiral_tight_boundary.pkl")
elif args.dataset == "toy_d60":
    data_source = os.path.join("data", "toy_d60_spiral.pkl")
with open(data_source, "rb") as data_file:
    data_pkl = pickle.load(data_file)
print(f"Data: {data_pkl['data'].shape}\nLabel: {data_pkl['label'].shape}")

opt.A = data_pkl["A"]
data = data_pkl["data"]
data_mean = data.mean(0, keepdims=True) # Only use src domains for normalization
data_std = data.std(0, keepdims=True)
data_pkl["data"] = (data - data_mean) / data_std  # normalize the raw data
datasets = [ToyDataset(data_pkl, i) for i in range(opt.num_domain)] # sub dataset for each domain
dataset = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=opt.batch_size)

model = Model(opt).to(opt.device)

# Train ERM model (super quick)
for epoch in range(args.epochs):
    model.learn(epoch, dataloader)

# Get embeddings from ERM model
# DG-15/60
train_x_seq = data_pkl["data"]
train_x_seq_t = to_tensor(train_x_seq)
train_x_seq_t = train_x_seq_t[None, :].to(torch.float)
# Feature
x_seq_feat = to_np(model.test_x_seq_feat(train_x_seq_t))
feat_size = x_seq_feat.shape[-1]
x_seq_feat = x_seq_feat.reshape((opt.num_domain, -1, feat_size))[opt.src_domain]
print("x_seq_feat", x_seq_feat.shape)

n_distributions = x_seq_feat.shape[0]
n_points_per_distribution = x_seq_feat.shape[1]

# Get distance matrix
if opt.diffusion == 0:
    # 1. Use mean/var to get the matrix
    dis_matrix = mean_var(x_seq_feat)
else:
    # 2. Use diffusion EMD to get the matrix
    dis_matrix = emd(x_seq_feat, feat_size, n_distributions, n_points_per_distribution)

print("distance matrix", dis_matrix)
print(dis_matrix.shape)

# Get percentile
dis_matrix_flat = dis_matrix.reshape(-1)
thres = np.percentile(dis_matrix_flat, opt.threshold)
# Unweighted graph
A = np.zeros_like(dis_matrix)
rule = (dis_matrix < thres) & (dis_matrix != 0)
A[rule] = 1

centrality = get_centrality(A)
# if centrality.sum() == 0.0:
#     centrality = np.ones_like(centrality)
# centrality /= centrality.sum()
centrality = np.around(centrality, 3)
print(list(centrality))