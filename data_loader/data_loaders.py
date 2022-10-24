from torchvision import datasets, transforms
from base import BaseDataLoader
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

from transformers import AutoTokenizer # huggingface 
MODEL_NAME = "bert-base-multilingual-cased"



class MyDataloader(BaseDataLoader):
    """
    KLUE SST Dataset
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=0, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = DF2DataSet(MODEL_NAME)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class DF2DataSet(Dataset):
    def __init__(self,MODEL_NAME = "bert-base-multilingual-cased", max_len=512, truncate=True):
        self.load_data()
        # label만 있는 데이터 생성
        self.label = self.df['gold_label']
        
        # label을 숫자로 변환하기 위한 label 모음
        self.get_labels = set(self.label)

        # 문자열을 tensor vector로 변환해주는는 tokenizer 준비
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # tokenizer로 두 문장 (premise, hypothesis)을 변환하여 입력 데이터 생성
        ## 두 문장을 string 타입의 list로 변환
        p_text = pd.Series(self.df['premise'], dtype="string").tolist()
        h_text = pd.Series(self.df['hypothesis'], dtype="string").tolist()

        self.input_data = self.tokenizer(text = p_text, text_pair = h_text,
                                          padding='max_length',
                                          truncation=truncate,
                                          return_tensors='pt',
                                          max_length=max_len)
    def load_data(self):
      valid_fn = '/content/KLUE/klue_benchmark/klue-nli-v1.1/klue-nli-v1.1_dev.json'
      self.df = pd.read_json(valid_fn)
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        item = {"inputs": self.input_data[idx], 
                # label을 id로 변환
                "labels": torch.tensor(self.get_labels.index(self.label.iloc[idx])),
                }
        return item