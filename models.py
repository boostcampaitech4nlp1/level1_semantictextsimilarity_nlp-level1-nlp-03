import torch
import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics
from transformers import AutoModelForSequenceClassification,ElectraForSequenceClassification, ElectraTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup


class STSModel(pl.LightningModule):
    def __init__(self, config) :
        super(STSModel, self).__init__()
        self.save_hyperparameters()

        self.model_path = config.model_path
        self.max_epochs = config.max_epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.warmup_ratio = config.warmup_ratio
        if config.model_type == 'roberta':
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=1)

        elif config.model_type == 'electra':
            self.model = ElectraForSequenceClassification.from_pretrained(self.model_path, num_labels=1)

        self.criterion = torch.nn.MSELoss()

    def forward(self, inputs):

        return self.model(input_ids = inputs['input_ids'],
                    attention_mask = inputs['attention_mask'],
                    return_dict=True)['logits']

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        data_len = len(self.trainer.datamodule.train_dataset)
        num_train_steps = int(data_len / self.batch_size * self.max_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_training_steps=num_train_steps, num_warmup_steps=num_warmup_steps)
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        y = batch['label']
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        y = batch['label']
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        val_pearson = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())
        self.log("val_pearson", val_pearson, prog_bar=True, on_step=False, on_epoch=True)

        return {'val_loss': loss, 'val_pearson': val_pearson}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        avg_pearson = torch.stack([output['val_pearson'] for output in outputs]).mean()
        print(f"\nval epoch loss : {avg_loss}\nval epoch pearson : {avg_pearson}")


    def predict_step(self, batch, batch_idx):
        logits = self(batch)

        return logits.squeeze()
