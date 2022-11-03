import pandas as pd

train_path = "../data/final_train.csv"
dev_path = "../data/final_val.csv"
test_path = '../data/final_test.csv'

lst = ['id','source','sentence_1','sentence_2','label','binary-label']



def clean():
    train_data = pd.read_csv(train_path)[lst]
    val_data = pd.read_csv(dev_path)[lst]
    test_data = pd.read_csv(test_path)[lst]

    
    train_data.to_csv('../data/clean_train.csv', index=False)
    val_data.to_csv('../data/clean_val.csv', index=False)
    test_data.to_csv('../data/clean_test.csv', index=False)

if __name__ == '__main__':
    clean()