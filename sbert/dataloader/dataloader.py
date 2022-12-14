import pandas as pd

from tqdm.auto import tqdm
from torch import nn
import pytorch_lightning as pl
import transformers
import torch
import torchmetrics
from utils import util

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs

        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return {'embed_1':torch.FloatTensor(self.inputs[idx][0]),
                    'embed_2':torch.FloatTensor(self.inputs[idx][1])}
        else:
            return {'embed_1':torch.FloatTensor(self.inputs[idx][0]),
                    'embed_2': torch.FloatTensor(self.inputs[idx][1]),
                    'label': torch.round(torch.FloatTensor([self.targets[idx]]),decimals=1).squeeze()}

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)



class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path,smodel):
        super().__init__()
        self.model_name = model_name
        self.smodel = SentenceTransformer(smodel)
        
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None


        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self,dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 임베딩을 문장단위로만 진행해줍니다.
            outputs = [self.smodel.encode(item[text_column]) for text_column in self.text_columns]
            data.append(outputs)
        return data
        

    def preprocessing(self, data): 
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            train_data = util.data_re(train_data)
            val_data = pd.read_csv(self.dev_path)
            val_data = util.data_re(val_data)
            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            predict_data = pd.read_csv(self.predict_path)
            predict_data = util.data_re(predict_data)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
