#! /bin/bash

trap "kill 0" EXIT

python cem_gym.py Walker2d-v2 -r 2 -l 25 -i 5
python cem_gym.py MyWalker2d-v2 -r 2 -l 25 -i 5

python cem_gym.py Ant-v2 -r 1 -l 25 -i 5
python cem_gym.py MyAnt-v2 -r 1 -l 25 -i 5

python cem_gym.py Humanoid-v2 -r 2 -l 25 -i 5
python cem_gym.py MyHumanoid-v2 -r 2 -l 25 -i 5

#CUDA_VISIBLE_DEVICES=1 python main.py --env half_cheetah --num_part 40 --ensemble_size 10 &
#CUDA_VISIBLE_DEVICES=2 python main.py --env half_cheetah --num_part 40 --ensemble_size 5  &

wait
