from omegaconf import OmegaConf
import argparse
import os

from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from util import make_sts_input_example
from sdataloader import STSDataset

from tqdm.auto import tqdm
from sentence_transformers import losses
from datetime import datetime
import math
def main(cfg):
    # 학습경로 
    train_data = pd.read_csv(cfg.train_path)
    val_data = pd.read_csv(cfg.val_path)
    # 데이터셋
    train_dataset = STSDataset(model,train_data,train_data['label'].to_numpy())
    val_dataset = STSDataset(model,val_data,val_data['label'].to_numpy())

    sts_train_examples = make_sts_input_example(train_dataset)
    sts_valid_examples = make_sts_input_example(val_dataset)

    # Train Dataloader
    train_dataloader = DataLoader(
        sts_train_examples,
        shuffle=True,
        batch_size=32, # 32 (논문에서는 16)
    )
    # Evaluator by sts-validation
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        sts_valid_examples,
        name="sts-dev",
    )
    model = SentenceTransformer(cfg.pretrained_model_name)
    # Use CosineSimilarityLoss
    train_loss = losses.CosineSimilarityLoss(model=model)
    # linear learning-rate warmup steps
    warmup_steps = math.ceil(len(sts_train_examples) * cfg.sts_num_epochs / cfg.train_batch_size * 0.1) #10% of train data for warm-up
    # Training
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=cfg.sts_num_epochs,
        evaluation_steps=int(len(train_dataloader)*0.1),
        warmup_steps=warmup_steps,
        output_path=cfg.sts_model_save_path,
        show_progress_bar = True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f'./config/{args.config}.yaml')
    main(cfg)
    if cfg.test:
        # 'output/hanseong_sts-klue-roberta-large-2022-11-01_18-23-15'
        sts_model_save_path = cfg.test.save_path
        model_new = SentenceTransformer(sts_model_save_path)
  
        test_data = pd.read_csv(cfg.test_path)


        vec1 = model_new.encode(test_data['sentence_1'], show_progress_bar=True, batch_size=32)
        vec2 = model_new.encode(test_data['sentence_2'], show_progress_bar=True, batch_size=32)

        test_data['target'] = [
            util.cos_sim(sent1, sent2).squeeze()*5
            for sent1, sent2 in tqdm(zip(vec1, vec2), total=len(test_data))
        ]