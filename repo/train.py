import argparse
import sys

import pandas as pd

from tqdm.auto import tqdm
from torch import nn
import pytorch_lightning as pl
import torch
from model import *
from dataloader import Dataloader

from utils import *

from pytorch_lightning.callbacks import ModelCheckpoint

pl.utilities.seed.seed_everything(1234)

def main():
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')

    parser.add_argument('--preprocess', default='base_preprocess')
    parser.add_argument('--model_arch', default='BaseModel')
    args = parser.parse_args()

    # dataloader와 model을 생성합니다.
    # wandb.init(project="SST", entity="roy_1201")
    # wandb.config = {
    #         "model_name": args.model_name,
    #         "learning_rate": args.learning_rate,
    #         "max_epoch": args.max_epoch,
    #         "batch_size": args.batch_size
    #         }
    dataloader = Dataloader(args.model_name, getattr(sys.modules[__name__], args.preprocess), args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = getattr(sys.modules[__name__], args.model_arch)(args.model_name, args.learning_rate)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.max_epoch, log_every_n_steps=1)

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    test_pearson = trainer.test(model=model, datamodule=dataloader)

    # test_pearson[0]['test_pearson']으로 점수 불러올 수 있음. 
    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, 'new_model2.pt')


if __name__ == '__main__':
    main()
