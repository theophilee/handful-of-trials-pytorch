#! /bin/bash

trap "kill 0" EXIT

python cem_gym.py Walker2d-v2 -r 2 -l 25 -i 5 --save
python cem_gym.py MyWalker2d-v2 -r 2 -l 25 -i 5 --save

python cem_gym.py Ant-v2 -r 1 -l 25 -i 5 --save
python cem_gym.py MyAnt-v2 -r 1 -l 25 -i 5 --save

python cem_gym.py Humanoid-v2 -r 2 -l 25 -i 5 --save
python cem_gym.py MyHumanoid-v2 -r 2 -l 25 -i 5 --save

python cem_gym.py InvertedPendulum-v2 -r 1 -l 25 -i 5 --save
python cem_gym.py MyInvertedPendulum-v2 -r 1 -l 25 -i 5 --save

wait
