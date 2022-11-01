#!/bin/bash

CONFIGS=("sample")

for (( i=0; i<${#CONFIGS[@]}; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]}
done
