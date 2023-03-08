#!/bin/bash

# DG-15 Physical Graph
PREFIX="--gpu_id 0 --dataset toy_d15 --epochs 20 --model TRO"
python3 main.py $PREFIX --learn 0

# DG-15 Data Graph
# Learn the graph and obtain the centrality
PREFIX="--gpu_id 0 --dataset toy_d15 --epochs 20"
python3 learn_graph.py $PREFIX

PREFIX="--gpu_id 0 --dataset toy_d15 --epochs 20 --model TRO"
python3 main.py $PREFIX --learn 1

# DG-60 Physical Graph
PREFIX="--gpu_id 0 --dataset toy_d60 --epochs 300 --model TRO"
python3 main.py $PREFIX --learn 0

# DG-60 Data Graph
# Learn the graph and obtain the centrality
PREFIX="--gpu_id 0 --dataset toy_d60 --epochs 100"
python3 learn_graph.py $PREFIX

PREFIX="--gpu_id 0 --dataset toy_d60 --epochs 300 --model TRO"
python3 main.py $PREFIX --learn 1