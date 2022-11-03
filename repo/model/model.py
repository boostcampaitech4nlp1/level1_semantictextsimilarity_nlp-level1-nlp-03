from torch import nn
import pytorch_lightning as pl
import transformers
import torch
import torchmetrics

class BaseModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = cfg.model.model_name
        self.lr = cfg.train.learning_rate

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=1, force_download=True)
        # Loss 계산을 위해 사용될 Loss 함수를 호출합니다. 
        self.loss_func = nn.SmoothL1Loss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)
        # wandb.log("train_loss", loss)
        return loss
    
    """
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.tensor([x for x in outputs]).mean()
        self.log("train_avg_loss", avg_train_loss)
        return avg_train_loss
    """

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        # wandb.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        # wandb.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    """
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x for x in outputs]).mean()
        self.log("val_avg_loss", avg_val_loss)
        return avg_val_loss
    """

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        # wandb.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

class SmoothL1Model(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_func = nn.SmoothL1Loss()




class MSEModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = cfg.model.model_name
        self.lr = cfg.train.learning_rate

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=1, force_download=True)
        # Loss 계산을 위해 사용될 Loss 함수를 호출합니다. 
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        eps = 1e-6
        loss = torch.sqrt(self.loss_func(logits, y.float())+eps)
        self.log("train_loss", loss)
        # wandb.log("train_loss", loss)
        return loss
    
    """
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.tensor([x for x in outputs]).mean()
        self.log("train_avg_loss", avg_train_loss)
        return avg_train_loss
    """

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        eps = 1e-6
        loss = torch.sqrt(self.loss_func(logits, y.float())+eps)
        self.log("val_loss", loss)
        # wandb.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        # wandb.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    """
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x for x in outputs]).mean()
        self.log("val_avg_loss", avg_val_loss)
        return avg_val_loss
    """

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        # wandb.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer