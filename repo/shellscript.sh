#! /bin/sh

# 실행방법: sh shellscript.sh

# train.py
for val in 0.1 0.01 0.001 0.0001
do
    python3 train.py --model_name='klue/roberta-small' --model_arch='BaseModel' --preprocess='base_preprocess' --learning_rate=$val
done