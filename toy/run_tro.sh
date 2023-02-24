#!/bin/bash
PREFIX="--gpu_id 0 --dataset toy_d15 --epochs 20 --model TRO"
python3 main.py $PREFIX