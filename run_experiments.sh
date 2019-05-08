#! /bin/bash

trap "kill 0" EXIT

python mpc_gym_true_dynamics_cmd_line.py MySwimmer-v2 -l 12 -r 9 -k 20 -e 10 &

wait
