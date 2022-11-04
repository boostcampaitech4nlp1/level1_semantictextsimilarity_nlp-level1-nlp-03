import torch
class STSDataset(torch.utils.data.Dataset):
    def __init__(self,model, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets
        self.model = model
        # self.path = path

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        sentences = self.inputs.iloc[idx]
        if len(self.targets) == 0:
            return {'sentence_1': sentences['sentence_1'],
                    'sentence_2' : sentences['sentence_2']}
        else:
            return {'sentence_1': sentences['sentence_1'],
                    'sentence_2' : sentences['sentence_2'],
                    'label': torch.FloatTensor([self.targets[idx]]).squeeze()
                    }
    def __len__(self):
        return len(self.inputs)