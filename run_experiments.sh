#! /bin/bash

trap "kill 0" EXIT

: '
python cem_comparison.py InvertedPendulum-v2 gaussian -r 1 -l 25 -i 5
python cem_comparison.py InvertedPendulum-v2 nonparametric -r 1 -l 25 -i 5
python cem_comparison.py InvertedPendulum-v2 gaussian -r 2 -l 25 -i 5
python cem_comparison.py InvertedPendulum-v2 gaussian -r 4 -l 25 -i 5

python cem_comparison.py Hopper-v2 gaussian -r 1 -l 25 -i 5
python cem_comparison.py Hopper-v2 nonparametric -r 1 -l 25 -i 5
python cem_comparison.py Hopper-v2 gaussian -r 2 -l 25 -i 5
python cem_comparison.py Hopper-v2 gaussian -r 4 -l 25 -i 5

python cem_comparison.py Walker2d-v2 gaussian -r 1 -l 25 -i 5
python cem_comparison.py Walker2d-v2 nonparametric -r 1 -l 25 -i 5
python cem_comparison.py Walker2d-v2 gaussian -r 2 -l 25 -i 5
python cem_comparison.py Walker2d-v2 gaussian -r 4 -l 25 -i 5

python cem_comparison.py Ant-v2 gaussian -r 1 -l 25 -i 5
python cem_comparison.py Ant-v2 nonparametric -r 1 -l 25 -i 5
python cem_comparison.py Ant-v2 gaussian -r 2 -l 25 -i 5
python cem_comparison.py Ant-v2 gaussian -r 4 -l 25 -i 5

python cem_comparison.py Humanoid-v2 gaussian -r 1 -l 25 -i 5
python cem_comparison.py Humanoid-v2 nonparametric -r 1 -l 25 -i 5
python cem_comparison.py Humanoid-v2 gaussian -r 2 -l 25 -i 5
python cem_comparison.py Humanoid-v2 gaussian -r 4 -l 25 -i 5
'

CUDA_VISIBLE_DEVICES=1 python main.py --env half_cheetah --num_part 40 --ensemble_size 10 &
CUDA_VISIBLE_DEVICES=2 python main.py --env half_cheetah --num_part 20 --ensemble_size 5  &

wait
