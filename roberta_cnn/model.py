import torch
import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics
from transformers import AutoModel

class STSModel(pl.LightningModule):
    def __init__(self, config) :
        super(STSModel, self).__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.model_path = config.model_path
        self.max_epochs = config.max_epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.model_type = config.model_type

        # 사용할 모델을 호출합니다.
        self.plm = AutoModel.from_pretrained(pretrained_model_name_or_path=self.model_path,force_download=True)
        
        self.cnn_layers1 = torch.nn.Conv1d(in_channels=768,out_channels=100,kernel_size=3)
        self.cnn_layers2 = torch.nn.Conv1d(in_channels=768,out_channels=100,kernel_size=4)
        self.cnn_layers3 = torch.nn.Conv1d(in_channels=768,out_channels=100,kernel_size=5)

        self.pooling_layers1 = torch.nn.MaxPool1d(kernel_size=510)
        self.pooling_layers2 = torch.nn.MaxPool1d(kernel_size=509)
        self.pooling_layers3 = torch.nn.MaxPool1d(kernel_size=508)

        self.fc_layer = torch.nn.Linear(300,1)

        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.SmoothL1Loss()
    def forward(self, inputs):

        x = self.plm(x).last_hidden_state.transpose(1,2)

        x1 = self.cnn_layers1(x)
        x2 = self.cnn_layers2(x)
        x3 = self.cnn_layers3(x)

        x1 = torch.relu(x1)
        x2 = torch.relu(x2)
        x3 = torch.relu(x3)

        x1_pooled = self.pooling_layers1(x1)
        x2_pooled = self.pooling_layers2(x2)
        x3_pooled = self.pooling_layers3(x3)

        x_cat = torch.cat([x1_pooled,x2_pooled,x3_pooled],dim=1).transpose(1,2).squeeze()

        x = self.fc_layer(x_cat)

        x = torch.relu(x)

        return x


    def configure_optimizers(self):
        plm_optimizer = torch.optim.AdamW(self.plm.parameters(), lr=self.lr)
        cnn_optimizer1 = torch.optim.AdamW(self.cnn_layers1.parameters(),lr=3e-5)
        cnn_optimizer2 = torch.optim.AdamW(self.cnn_layers2.parameters(),lr=3e-5)
        cnn_optimizer3 = torch.optim.AdamW(self.cnn_layers3.parameters(),lr=3e-5)
        liner_optimizer = torch.optim.AdamW(self.fc_layer.parameters(),lr = 3e-5)
        return plm_optimizer,cnn_optimizer1,cnn_optimizer2,cnn_optimizer3,liner_optimizer

    def training_step(self, batch, batch_idx,optimizer_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()


