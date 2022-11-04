import torch
import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics
from transformers import AutoModelForSequenceClassification,ElectraForSequenceClassification, AutoModel
from transformers.optimization import get_cosine_schedule_with_warmup
from utils import mean_pooling


class STSModel(pl.LightningModule):
    def __init__(self, config) :
        super(STSModel, self).__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.model_path = config.model_path
        self.max_epochs = config.max_epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.warmup_ratio = config.warmup_ratio
        self.model_type = config.model_type

        # source embedding
        if config.source_emb:
            # electra -> (b , l , emb) -> (b , emb)
            self.model = AutoModel.from_pretrained(self.model_path)

            # source emb
            # (1, 768)
            self.source_emb = torch.nn.Embedding(2, self.model.embeddings.word_embeddings.weight.size(-1))

            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(self.model.embeddings.word_embeddings.weight.size(-1), self.model.embeddings.word_embeddings.weight.size(-1)//2),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.config.dropout),
                torch.nn.Linear(self.model.embeddings.word_embeddings.weight.size(-1)//2, 1)
            )


        else:    
            if config.model_type == 'roberta':
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=1)

            elif config.model_type == 'electra':
                self.model = ElectraForSequenceClassification.from_pretrained(self.model_path, num_labels=1)

        self.criterion = torch.nn.MSELoss()

    def forward(self, inputs):
        if self.config.source_emb:
            
            # B, 768
            model_logits = self.model(input_ids = inputs['input_ids'],
                    attention_mask = inputs['attention_mask'],
                    token_type_ids = inputs['token_type_ids'],
                    return_dict=True)['last_hidden_state'][:, 0, :]
            # b, 768
            source = self.source_emb(inputs['source'])

            logits = model_logits + source.squeeze()
            outputs = self.output_layer(logits)
            return outputs
            


        else:
            return self.model(input_ids = inputs['input_ids'],
                    attention_mask = inputs['attention_mask'],
                    token_type_ids = inputs['token_type_ids'],
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


class Model(pl.LightningModule):
    def __init__(self, model_path, max_epochs, lr, warmup_ratio, batch_size) :
        super(Model, self).__init__()
        self.save_hyperparameters()
        
        
        self.model_path = model_path
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_ratio = warmup_ratio

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=1)

        self.criterion = torch.nn.MSELoss()

    def forward(self, inputs):
        
        return self.model(input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                token_type_ids = inputs['token_type_ids'],
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

class StackingModel(pl.LightningModule):
    def __init__(self, config) :
        super(StackingModel, self).__init__()
        self.save_hyperparameters()
        self.lr = config.lr
        self.max_epochs = config.max_epochs
        self.warmup_ratio = config.warmup_ratio
        self.dropout = config.dropout
        self.batch_size = config.batch_size

        self.layers = torch.nn.Sequential(
                            torch.nn.Linear(4, 2, bias=config.bias),
                            torch.nn.Dropout(p=self.dropout),
                            torch.nn.GELU(),
                            torch.nn.Linear(2, 1, bias=config.bias)
                        )

        self.criterion = torch.nn.MSELoss()

    def forward(self, inputs):
        outputs = self.layers(inputs)
        
        return outputs


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        data_len = len(self.trainer.datamodule.train_dataset)
        num_train_steps = int(data_len / self.batch_size * self.max_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_training_steps=num_train_steps, num_warmup_steps=num_warmup_steps)
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        logits = self(batch['inputs'])
        y = batch['label']
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['inputs'])
        y = batch['label']
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        print(f"\nval epoch loss : {avg_loss}\n")


    def predict_step(self, batch, batch_idx):
        logits = self(batch['inputs'])

        return logits.squeeze()