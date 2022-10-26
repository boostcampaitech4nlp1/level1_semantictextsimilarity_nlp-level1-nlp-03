import pandas as pd
import os
from tqdm.auto import tqdm
import transformers
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
class BaseDataset(Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        if len(self.targets) == 0:  #test
            return torch.tensor(self.inputs[idx])
        else:                       #train
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class BaseDataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, path, max_length, num_workers):   #path 통합함, max_legth, numworkers 추가함
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_length=max_length
        self.num_workers=num_workers

        self.train_path = os.path.join(path,"train.csv")
        self.dev_path = os.path.join(path,"dev.csv")
        self.test_path = os.path.join(path,"dev.csv")
        self.predict_path = os.path.join(path,"test.csv")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, max_length=self.max_length)       #AutoTokenizer안쓰면 변경해야함
        self.target_columns = ['label']
        self.delete_columns = ['id']                                                                    #버릴 컬럼 필요에 따라 변경 가능 ,'source','binary-label' 추가하기(test엔 없다.)
        self.text_columns = ['sentence_1', 'sentence_2']                                                    #필요에 따라 변경 가능

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)
        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.                                                                        #데이터 전처리 추가할것
        inputs = self.tokenizing(data)

        return inputs, targets
    
    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터
            train_data = pd.read_csv(self.train_path)
            train_inputs, train_targets = self.preprocessing(train_data)
            self.train_dataset = BaseDataset(train_inputs, train_targets)
            # 검증데이터
            val_data = pd.read_csv(self.dev_path)
            val_inputs, val_targets = self.preprocessing(val_data)
            self.val_dataset = BaseDataset(val_inputs, val_targets)

        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = BaseDataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = BaseDataset(predict_inputs, [])

    def train_dataloader(self):
        # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)
