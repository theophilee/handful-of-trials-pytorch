#! /bin/bash

trap "kill 0" EXIT

#python mpc_gym_true_dynamics_cmd_line.py MyHalfCheetah-v2 -l 12 -r 4 -k 60 -e 10
#python mpc_gym_true_dynamics_cmd_line.py MySwimmer-v2 -l 16 -r 9 -k 20 -e 10
#python mpc_gym_true_dynamics_cmd_line.py MyCartpole-v0 -l 12 -r 4 -k 50 -e 10

CUDA_VISIBLE_DEVICES=1 python main.py --env half_cheetah --activation relu &
CUDA_VISIBLE_DEVICES=2 python main.py --env half_cheetah --activation swish &

wait