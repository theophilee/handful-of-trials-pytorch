#! /bin/bash

trap "kill 0" EXIT

CUDA_VISIBLE_DEVICES=0 python main.py --activation swish &
CUDA_VISIBLE_DEVICES=2 python main.py --activation relu &

#python cem_gym.py MyWalker2d-v2 -r 2 -l 25 -i 5 -e 5 --save
#python cem_gym.py MyAnt-v2 -r 1 -l 25 -i 5 -e 5 --save
#python cem_gym.py MyHumanoid-v2 -r 2 -l 25 -i 5 -e 5 --save
#python cem_gym.py MyInvertedPendulum-v2 -r 1 -l 25 -i 5 -e 5 --save

wait