import argparse
import sys
import datetime

import pandas as pd

from tqdm.auto import tqdm
from torch import nn
import pytorch_lightning as pl
import torch
from model import *
from dataloader import Dataloader

from utils import *

from pytorch_lightning.callbacks import ModelCheckpoint

from omegaconf import OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger

pl.utilities.seed.seed_everything(1234)
torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='sample')
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f'./config/{args.config}.yaml')

    wandb.login()
    wandb_logger = WandbLogger(name=cfg.model.saved_name, project='aistages')

    dataloader = Dataloader(cfg.model.model_name, getattr(sys.modules[__name__], cfg.custom.preprocess), 
                            getattr(sys.modules[__name__], cfg.data.augmentation),
                            cfg.train.batch_size, cfg.data.shuffle, 
                            cfg.path.train_path, cfg.path.dev_path,
                            cfg.path.test_path, cfg.path.predict_path)
    model = getattr(sys.modules[__name__], cfg.custom.model_arch)(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath='./save/ckpt/',
        filename=cfg.model.saved_name+'-{epoch}-{val_pearson:.4f}',
        mode='max',
        monitor='val_pearson',
        save_top_k=3,
        save_last=True)

    trainer = pl.Trainer(gpus=cfg.train.gpus, max_epochs=cfg.train.max_epoch, 
                        logger=wandb_logger, log_every_n_steps=cfg.train.logging_step,
                        callbacks=[checkpoint_callback])

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    
    torch.save(model, './save/model/' + cfg.model.saved_name + '.pt')

    # test_pearson[0]['test_pearson']으로 점수 불러올 수 있음. 
    # 학습이 완료된 모델을 저장합니다.
    ckpt = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    trainer.test(model=ckpt, datamodule=dataloader)
    


if __name__ == '__main__':
    main()
