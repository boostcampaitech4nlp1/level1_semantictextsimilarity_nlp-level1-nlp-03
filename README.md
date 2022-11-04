# Results

## ELECTRA

<center><img src="./imgs/electra_loss.png" width=800></center>
<br>
<center><img src="./imgs/electra_pearson.png" width=800></center>

## RoBERTa + CNN
klue/RoBERTa (base)
- epochs 15, batch_size 16
- val_loss = 0.172, val_pearson = 0.921
<center><img src="./imgs/roberta-base.png" width=800></center>
klue/RoBERTa + CNN_layers

- epochs 15, batch_size 16
- val_loss = 0.154, val_pearson = 0.926
<center><img src="./imgs/rbcnn.png" width=800></center>

## sRoBERTa_Large
klue/RoBERTa (base)
- epochs 200, batch_size 32
- val_loss = 0.106, val_pearson = 0.9754
<center><img src="./imgs/sroberta-large.png" width=800></center>
