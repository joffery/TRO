#!/bin/bash

# DG-15 Physical Graph
python3 main.py --dataset toy_d15 --learn 0 --model TRO

# DG-15 Data Graph
# Learn the graph and obtain the centrality
python3 learn_graph.py --dataset toy_d15

python3 main.py --dataset toy_d15 --learn 1 --model TRO

# DG-60 Physical Graph
python3 main.py --dataset toy_d60 --learn 0 --model TRO

# DG-60 Data Graph
# Learn the graph and obtain the centrality
python3 learn_graph.py --dataset toy_d60

python3 main.py $PREFIX --dataset toy_d60 --learn 1 --model TRO