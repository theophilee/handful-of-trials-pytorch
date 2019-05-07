#! /bin/bash

trap "kill 0" EXIT

python mpc_gym_true_dynamics.py MyCartpole-v0 &
python mpc_gym_true_dynamics.py MyHalfCheetah-v2 &

wait
