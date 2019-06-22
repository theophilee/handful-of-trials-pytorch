#! /bin/bash

trap "kill 0" EXIT

CUDA_VISIBLE_DEVICES=1 python main.py --env cartpole &
#CUDA_VISIBLE_DEVICES=2 python main.py --env half_cheetah &

wait