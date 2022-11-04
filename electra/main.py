import argparse
from models import STSModel
from dataloader import STSDataModule
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning as pl
import os
import pandas as pd
import torch
from config import Config

random_initiallization = 42
pl.seed_everything(random_initiallization)

def main(args):
    
    
    if args.checkpoint:
        # Config (test or continuous)
        ckpt_path = os.path.join('logs/models', args.checkpoint + '.ckpt')
        model_info = torch.load(ckpt_path)
        config = model_info['hyper_parameters']['config']
        
    else:
        config_path = args.config
        # Config (train)
        config = Config(config_path)

    dm = STSDataModule(tok_path=config.tok_path,
                            train_path=config.train_path,
                            valid_path=config.val_path,
                            test_path=config.test_path,
                            batch_size=config.batch_size)

    if args.test:
        print('predict')
        load_model = STSModel.load_from_checkpoint(checkpoint_path=ckpt_path)
        trainer = pl.Trainer(accelerator='gpu', devices=1)
        predictions = trainer.predict(model=load_model, datamodule=dm)
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))
        output = pd.read_csv('data/sample_submission.csv')
        output['target'] = predictions
        output.to_csv(config.output_path, index=False)

    else:
        # prefix = config.train_path[config.train_path.rfind('/')+1:config.train_path.rfind('.')]
        prefix = config.model_type
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs', name=f'tb_logs/{prefix}')
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                    dirpath=f'logs/models/{prefix}',
                                    monitor='val_pearson',
                                    mode='max',
                                    filename='{epoch:02d}-{val_pearson:.3f}-{val_loss:.3f}',
                                    save_last=True,
                                    verbose=True,
                                    save_top_k=3
                                    )

        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=config.max_epochs, logger=tb_logger, callbacks=[checkpoint_callback], log_every_n_steps=1)
        if args.checkpoint:
            print("continous train")
            trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.add_epochs + config.max_epochs, logger=tb_logger, callbacks=[checkpoint_callback], resume_from_checkpoint=ckpt_path, log_every_n_steps=1)
            model = STSModel.load_from_checkpoint(checkpoint_path=ckpt_path)

            model.max_epochs = args.add_epochs
            model.lr = args.add_lr

        else:
            print('train')
            model = STSModel(config)

        trainer.fit(model, dm)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='electra_config.json')
    parser.add_argument('--test', '-t', action="store_true", default=False)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--add_epochs', type=int)
    parser.add_argument('--add_lr', type=float)
    args = parser.parse_args()
    # config_path = './config.json'
    main(args)