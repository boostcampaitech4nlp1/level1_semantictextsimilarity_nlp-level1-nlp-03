import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl


### TODO : path, filename 정해지면 수정
# from PATH import CUSTOMDATAMODULE as DataModule
# from PATH import CUSTOMMODEL as Model
# from PATH import CUSTOMTRAINER as Trainer
# from PATH import CUSTOMPREPROCESS as preprocess


def inference(args):
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
    
    # Inference
    model = torch.load(args.best_model_path)
    predictions = trainer.predict(model=model,
                                  datamodule=datamodule)
    
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions

    output_name = args.best_model_path[:-3] + '.csv'
    output.to_csv('output.csv', index=False)
    

if __name__ == '__main__':
    ### TODO : best model의 config값 불러와서 argument로 지정
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_model_path', default='model.pt')

    args = parser.parse_args(args=[])

    inference(args)
    