from torch import nn
import pytorch_lightning as pl
import transformers
import torch
class STSModel(pl.LightningModule):
    def __init__(self, config):
        super(STSModel,self).__init__()
        self.save_hyperparameters()

        self.model_path = config.model.model_name
        self.max_epoch = config.train.max_epoch
        self.batch_size = config.train.batch_size
        self.lr = config.train.learning_rate
        self.warmup_ratio = config.model.warmup_ratio


        ###### model fc #####
        self.fc1 = nn.Linear(1, 64, bias=True)
        self.fc2 = nn.Linear(64, 128, bias=True)
        self.fc3 = nn.Linear(128, 64, bias=True)
        self.fc4 = nn.Linear(64, 1, bias=False)

        self.dropout = nn.Dropout(0.2)
        ####### cnn model #####
        self.cnn_list = nn.ModuleList([torch.nn.Conv1d(in_channels=1,out_channels=100,kernel_size=1024) for i in range(3)])

        self.pooling_list = nn.ModuleList([torch.nn.AvgPool1d(kernel_size=2) for i in range(3)])
        self.sim_func = util.cos_sim

        # Loss 계산을 위해 사용될 SmoothL1Loss를 호출합니다.
        self.loss_func = nn.SmoothL1Loss()

    def forward(self, input_embeds):
        embed_1,embed_2 = input_embeds['embed_1'], input_embeds['embed_2']
        # pooling1 = self.conv_mean_pooler(embed_1,self.cnn_list,self.pooling_list)
        # pooling2 = self.conv_mean_pooler(embed_2,self.cnn_list,self.pooling_list)
        # x = torch.cat([pooling1,pooling2],dim=1)
        # x = F.relu(self.fc4(x))
        x = self.sim_func(embed_1,embed_2).diag()
        x = F.relu(self.fc1(x.reshape(-1,1)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.dropout(self.fc4(x)))
                         

        return x
    def conv_mean_pooler(self,model_output,conv1d,pool):
        token_embeddings = model_output.unsqueeze(1)
        x1 = torch.nn.functional.relu(conv1d[0](token_embeddings))
        x2 = torch.nn.functional.relu(conv1d[1](token_embeddings))
        x3 = torch.nn.functional.relu(conv1d[2](token_embeddings))
        # print('x1',x1.shape)
        x1_pooled = x1.squeeze(-1)
        x2_pooled = x2.squeeze(-1)
        x3_pooled = x3.squeeze(-1)

        x_cat = torch.cat([x1.squeeze(-1),x2.squeeze(-1),x3.squeeze(-1)],dim=1)
        x = F.relu(self.fc1(x_cat))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        # print('x_cat',x.shape)
        return x

    def training_step(self, batch, batch_idx):
        y = batch['label']
        del batch['label']
        x = batch
        logits = self(x)
        loss = self.loss_func(logits.squeeze(), y.float())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['label']
        del batch['label']
        x = batch
        logits = self(x)
        loss = self.loss_func(logits.squeeze(), y.float())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        val_pearson = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())
        self.log("val_pearson", val_pearson, prog_bar=True, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'val_pearson': val_pearson}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        avg_pearson = torch.stack([output['val_pearson'] for output in outputs]).mean()
        print(f"\nval epoch loss : {avg_loss}\nval epoch pearson : {avg_pearson}")
        


    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        data_len = len(self.trainer.datamodule.train_dataset)
        num_train_steps = int(data_len / self.batch_size * self.max_epoch)
        num_warmup_steps = int(num_train_steps * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_training_steps=num_train_steps, num_warmup_steps=num_warmup_steps)
        
        return [optimizer], [scheduler]