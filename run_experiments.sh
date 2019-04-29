#! /bin/bash

trap "kill 0" EXIT

CUDA_VISIBLE_DEVICES=0 python main.py --env "cartpole" --logdir "logs/cartpole" &
CUDA_VISIBLE_DEVICES=1 python main.py --env "pusher" --logdir "logs/pusher" &

wait