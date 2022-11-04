from omegaconf import OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger
import argparse
import os
import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from dataloader import Dataloader
from model import STSModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f'./config/{args.config}.yaml')
    wandb.login()
    wandb_logger = WandbLogger(name='roy_1201', project='SST')

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(cfg.model.model_name, cfg.train.batch_size, cfg.data.shuffle, cfg.path.train_path, cfg.path.dev_path,
                            cfg.path.test_path, cfg.path.predict_path,cfg.model.smodel)
    
    model = Model(args.model_name, args.learning_rate)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(gpus=cfg.train.gpus, max_epochs=cfg.train.max_epoch, logger=wandb_logger, log_every_n_steps=cfg.train.logging_step)


    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, f'{args.model_name}_model.pt')


if __name__ == '__main__':
    main()
