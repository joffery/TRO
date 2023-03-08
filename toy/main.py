import torch
import numpy as np
import random
import pickle
import argparse
import os

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from data_loader.data_loader import ToyDataset, SeqToyDataset
from utils.utils import *

parser = argparse.ArgumentParser(description='Topology-Aware Robust Optimization for OOD Generalization')
parser.add_argument('--dataset', default='toy_d15', type=str, help='toy_d15, toy_d60, weather')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--model', default='ERM', type=str, help='ERM, IRM, DRO, TRO')
parser.add_argument('--batch_size', default=10, type=int, help='mini-batch size (default: 10)')
parser.add_argument('--gpu_id', default=1, type=int, help='gpu id')

# Partial
parser.add_argument('--learn', default=1, type=int, help='data graph (1) or physical (0)')
parser.add_argument('--partial', default=0, type=int, help='source (1) or source + target (0), only for physical graph')

# IRM
parser.add_argument('--irm_penal', default=1e-1, type=float, help='irm penalty coefficient')

cudnn.benchmark = True

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 on stackoverflow
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

if args.dataset == "toy_d15":
    from configs.dg15 import opt
elif args.dataset == "toy_d60":
    from configs.dg60 import opt

# random seed
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

opt.model = args.model
opt.dataset = args.dataset

print("model: {}".format(opt.model))
print("dataset: {}".format(opt.dataset))

if opt.model == "ERM":
    from model.model import ERM as Model
elif opt.model == "DRO":
    from model.model import DRO as Model
elif opt.model == "IRM":
    from model.model import IRM as Model
    opt.irm_penal = args.irm_penal
elif opt.model == "TRO":
    from model.model import TRO as Model

# Important params
opt.num_epoch = args.epochs
opt.batch_size = args.batch_size

opt.learn = args.learn
opt.partial = args.partial

if args.dataset == "toy_d15":
    opt.num_domain = 15
    # the specific source and target domain:
    opt.src_domain = [0, 12, 3, 4, 14, 8] #Corresponds to 1-6 in Figure 2 (a)
elif args.dataset == "toy_d60":
    opt.num_domain = 60
    # the specific source and target domain:
    opt.src_domain = list(range(6))

opt.num_source = len(opt.src_domain)
opt.num_target = opt.num_domain - opt.num_source

if args.dataset == "toy_d15":
    data_source = os.path.join("data", "toy_d15_spiral_tight_boundary.pkl")
elif args.dataset == "toy_d60":
    data_source = os.path.join("data", "toy_d60_spiral.pkl")
else:
    raise NotImplementedError("Dataset not implemented. Please try toy_d15 or toy_d60!")
    
with open(data_source, "rb") as data_file:
    data_pkl = pickle.load(data_file)
print(f"Data: {data_pkl['data'].shape}\nLabel: {data_pkl['label'].shape}")

# set up experiment directory
opt.outf = setup_experiment(args, opt)

# build dataset
opt.A = data_pkl["A"] # physical graph's adjacent matrix
data = data_pkl["data"]

# dataloader
datasets = [ToyDataset(data_pkl, i) for i in range(opt.num_domain)] # sub dataset for each domain
dataset = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=opt.batch_size)

model = Model(opt).to(opt.device)

# train
for epoch in range(opt.num_epoch):
    model.learn(epoch, dataloader)

    if (epoch + 1) % opt.save_interval == 0 or (epoch + 1) == opt.num_epoch:
        model.save(model.model_path)
    if (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.num_epoch:
        model.test(epoch, dataloader)