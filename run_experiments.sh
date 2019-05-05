#! /bin/bash

trap "kill 0" EXIT

for repeat in 3 4 5 6
do

    python mpc_gym_true_dynamics.py MyHalfCheetah-v2 -r $repeat &

done

wait