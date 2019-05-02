#! /bin/bash

trap "kill 0" EXIT

CUDA_VISIBLE_DEVICES=0 python main.py --ensemble-size 1 &
CUDA_VISIBLE_DEVICES=1 python main.py --ensemble-size 2 &
CUDA_VISIBLE_DEVICES=2 python main.py --ensemble-size 5 &

wait