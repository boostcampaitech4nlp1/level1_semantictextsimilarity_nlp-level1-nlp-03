import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

### TODO : path, filename 정해지면 수정
# from PATH import CUSTOMDATAMODULE as DataModule
# from PATH import CUSTOMMODEL as Model
# from PATH import CUSTOMTRAINER as Trainer
# from PATH import CUSTOMPREPROCESS as preprocess



# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

seed_everything(SEED, workers=True) # pl seed



def train(args):
    ### TODO : preprocess 함수도 인자로 넘겼으면..
    datamodule = DataModule(args.model_name, 
                            args.batch_size, args.shuffle, 
                            args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
                        
    model = Model(args.model_name, 
                  args.learning_rate)

    trainer = Trainer(gpus=1, 
                      max_epochs=arg.max_epoch,
                      log_every_n_steps=1)
    
    trainer.fit(model=model, datamodule=datamodule)
    test_dict = trainer.test(model=model, datamodule=datamodule) # val로 테스트함. 


    # 모델 이름 자동 생성
    # 형식: val 성능 + 모델이름 + 저장시각 + 실험자.pt
    saved_model_path = './save/model/'
    test_pearson = test_dict[0]['test_pearson']

    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d_%H:%M:%S')
    saved_model_path += '+'.join([str(test_pearson), 
                                    args.model_name, 
                                    now_str,
                                    args.expreimenter,
                                    '.pt'])
    
    torch.save(model, saved_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')

    ### TODO : custom argument
    parser.add_argument('--preprocess', default="")
    parser.add_argument('--experimenter', default='bc-nlp-03')
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--loss_func', default='L1Loss')
    parser.add_argument('--num_workers', default=4)

    args = parser.parse_args(args=[])

    train(args)

"""
def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                    config=config,
                    device=device,
                    data_loader=data_loader,
                    valid_data_loader=valid_data_loader,
                    lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
"""