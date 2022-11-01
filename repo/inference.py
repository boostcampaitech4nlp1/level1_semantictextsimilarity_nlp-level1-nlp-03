import argparse
import sys

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from dataloader import *

from utils import *

from pytorch_lightning.callbacks import ModelCheckpoint

from omegaconf import OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='sample')
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f'./config/{args.config}.yaml')

    dataloader = Dataloader(cfg.model.model_name, getattr(sys.modules[__name__], cfg.custom.preprocess), 
                            getattr(sys.modules[__name__], cfg.data.augmentation),
                            cfg.train.batch_size, cfg.data.shuffle, 
                            cfg.path.train_path, cfg.path.dev_path,
                            cfg.path.test_path, cfg.path.predict_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath='./save/ckpt/',
        filename=cfg.model.saved_name+'-{epoch}-{val_pearson:.2f}')

    trainer = pl.Trainer(gpus=cfg.train.gpus, max_epochs=cfg.train.max_epoch, 
                        log_every_n_steps=cfg.train.logging_step,
                        callbacks=[checkpoint_callback])

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = torch.load('./save/model/' + cfg.model.saved_name + '.pt')
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('../data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('./save/output/' + cfg.model.saved_name + '.csv', index=False)

    # scatterplot
    scatterplot(model, trainer, dataloader, cfg.model.saved_name + '.png')


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    main()