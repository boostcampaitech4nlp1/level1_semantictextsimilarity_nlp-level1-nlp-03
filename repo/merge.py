import pandas as pd
import numpy as np


train_path = "../data/final_train.csv"
dev_path = "../data/final_val.csv"
test_path = '../data/final_test.csv'

lst = ['id','source','sentence_1','sentence_2','label','binary-label']

def train_val():
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(dev_path)

    cat = pd.concat([train_data, val_data])[lst]
    cat.to_csv('../data/final_train_val.csv', index=False)


def train_val_test():
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(dev_path)
    test_data = pd.read_csv(test_path)

    cat = pd.concat([train_data, val_data, test_data])[lst]
    cat.to_csv('../data/final_train_val_test.csv', index=False)




if __name__ == "__main__":
    train_val()
    train_val_test()