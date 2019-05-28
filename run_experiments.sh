#! /bin/bash

trap "kill 0" EXIT

CUDA_VISIBLE_DEVICES=2 python main.py --env half_cheetah --hid_features 200,200

#python cem_comparison.py MySwimmer-v2 gaussian -l 30 -p 1000 -i 5
#python cem_comparison.py MySwimmer-v2 gaussian -l 20 -p 1000 -i 5
#python cem_comparison.py MySwimmer-v2 gaussian -l 30 -p 1000 -i 10
#python cem_comparison.py MySwimmer-v2 gaussian -l 20 -p 1000 -i 10

wait
