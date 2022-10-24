# Overview
## 구성
- model.py
    - BaseModel을 참조해 객체를 생성하면 됨
    ```python
    from transformers import AutoTokenizer, AutoModel
    
    from base import BaseModel
    class OurModel(BaseModel):
        def __init__(self, name, age):
            super(OurModel,self).__init__()
            self.tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1",use_fast=False,                       add_prefix_space=True)
            self.model = AutoModel.from_pretrained("skt/kobert-base-v1")
        def forward(self,x):
            # overiding
    ```
- metric.py
    ```python
        def score_function(real, pred):
            return f1_score(real, pred, average="weighted")
    ```
- loss.py
    ```python
    from transformers import AdamW
    import torch.nn as nn
    optimizer = AdamW(model.parameters())
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,num_training_steps = total_steps)
    criterion = nn.CrossEntropyLoss().to(device)

    ```
