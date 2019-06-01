#! /bin/bash

trap "kill 0" EXIT

CUDA_VISIBLE_DEVICES=1 python main.py --env swimmer &
CUDA_VISIBLE_DEVICES=2 python main.py --env pusher &
CUDA_VISIBLE_DEVICES=2 python main.py --env cartpole &

wait
