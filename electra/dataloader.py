import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from utils import remove_word

class STSDataset(Dataset):
    def __init__(self, path, tokenizer, mode) -> None:
        super(STSDataset, self).__init__()
        self.mode = mode
        self.data = pd.read_csv(path)
        self.data.replace('', np.nan, inplace=True)
        self.data.dropna(axis=0, inplace=True)
        self.sources = self.data['source'].tolist()

        if self.mode != 'Test':
            self.labels = self.data['label'].values.tolist()
        self.columns = ['sentence_1', 'sentence_2']
        self.tokenizer = tokenizer
        self.max_len = self.tokenizer.model_max_length
        self.sep_tok = self.tokenizer.sep_token if self.tokenizer.sep_token else '[SEP]'
        self.preprocessing()
        
        print(f"length of the {self.mode} Data : {len(self.data)}")

    def preprocessing(self):
        data = []
        for idx, rows in tqdm(self.data[self.columns].iterrows(), desc=f"tokenizing...({self.mode})"):
            # remove_word(rows)
            text = '[SEP]'.join(rows.tolist())

            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs)
            
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'Test':
            return {
            'input_ids': torch.LongTensor(self.data[index]['input_ids']),
            'attention_mask': torch.LongTensor(self.data[index]['attention_mask']),
            'token_type_ids': torch.LongTensor(self.data[index]['token_type_ids']),
            'source': torch.LongTensor([self.sources[index]])
            }

        return {
            'input_ids': torch.LongTensor(self.data[index]['input_ids']),
            'attention_mask': torch.LongTensor(self.data[index]['attention_mask']),
            'token_type_ids': torch.LongTensor(self.data[index]['token_type_ids']),
            'source': torch.LongTensor([self.sources[index]]),
            'label': torch.FloatTensor([self.labels[index]])
        }

class STSDataModule(pl.LightningDataModule):
    def __init__(self, tok_path, train_path, valid_path, test_path, batch_size=8):
        super(STSDataModule, self).__init__()
        self.tok_path = tok_path
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.tok_path)

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_dataset = STSDataset(self.train_path, self.tokenizer, mode='Train')
            self.valid_dataset = STSDataset(self.valid_path, self.tokenizer, mode='Valid')

        else:
            self.test_dataset = STSDataset(self.test_path, self.tokenizer, mode='Test')

    
    def train_dataloader(self) :
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class StackingDataset(Dataset):
    def __init__(self, path, mode) -> None:
        super(StackingDataset, self).__init__()
        self.data = pd.read_csv(path)
        self.mode = mode

        print(f'{mode} : {len(self.data)}')


    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if self.mode == 'Test':
            return {
                'inputs': torch.FloatTensor(self.data.iloc[index][['electra','xlm','klue','cnn']])
            }

        return {'inputs': torch.FloatTensor(self.data.iloc[index][['electra','xlm_pred','klue','rbcnn']]),
                'label': torch.FloatTensor([self.data.iloc[index]['label']])}


class StackingDataModule(pl.LightningDataModule):
    def __init__(self, train_path, valid_path, test_path, batch_size=8):
        super(StackingDataModule, self).__init__()
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.batch_size = batch_size

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_dataset = StackingDataset(self.train_path, mode='Train')
            self.valid_dataset = StackingDataset(self.valid_path, mode='Valid')

        else:
            self.test_dataset = StackingDataset(self.test_path, mode='Test')

    
    def train_dataloader(self) :
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
