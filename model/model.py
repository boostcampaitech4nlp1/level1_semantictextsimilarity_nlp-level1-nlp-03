from base import base_model
import torch
import torch.nn as nn
import transformers.AutoTokenizer as AutoToeknizer


class Transformers(base_model):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = AutoToeknizer('bert-uncase')