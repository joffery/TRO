import pickle
import numpy as np
import torch
import math
import os
import json
from datetime import datetime

def to_np(x):
    return x.detach().cpu().numpy()

def to_tensor(x, device="cuda"):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x

def flat(x):
    n, m = x.shape[:2]
    return x.reshape(n * m, *x.shape[2:])

def read_pickle(name):
    with open(name, "rb") as f:
        data = pickle.load(f)
    return data

def write_pickle(data, name):
    with open(name, "wb") as f:
        pickle.dump(data, f)

def setup_experiment(args, opt):
    # log file
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    outfolder = os.path.join('runs', args.dataset, args.model, current_time)

    # make directories
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    # TODO: copy config file into the directory
    with open(os.path.join(outfolder, 'config.json'), 'w') as outfile:
        json.dump(opt, outfile, indent=4)

    return outfolder

def projection_simplex(v, z=1):
    """
    Old implementation for test and benchmark purposes.
    The arguments v and z should be a vector and a scalar, respectively.
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def arr2str(arr):
    return [str(i) for i in arr]

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.asarray([qx, qy])