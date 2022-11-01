#!/bin/bash

CONFIGS=("sample" "hanspell")

for (( i=0; i<${#CONFIGS[@]}; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]}
    python3 inference.py --config ${CONFIGS[$i]}
done