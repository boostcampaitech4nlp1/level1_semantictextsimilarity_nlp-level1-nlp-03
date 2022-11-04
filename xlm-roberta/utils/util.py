import seaborn as sns
import pandas as pd

from hanspell import spell_checker
from soynlp.normalizer import *

import torch

import re

# from .eda import EDA



### PREPROCESSING

def base_preprocess(data):
    """
    preprocess 함수의 베이스. 

    inputs:
        data: pandas.DataFrame
        데이터 프레임 중 sentence_1, sentence_2만 수정해서 그대로 내보내야 함. 
        텍스트 컬럼: ['sentence_1', 'sentence_2']

    outputs: 
        data_output: pandas.DataFrame
    """
    print('='*10)
    print('base preprocess')
    print('='*10)
    return data

def hanspell(data):
    newdata = data.copy()
    for i, (s1, s2) in enumerate(zip(data['sentence_1'], data['sentence_2'])):
        try:
            s1_check = spell_checker.check(s1).checked
        except:
            s1_check = s1
            print("ERROR: " + s1)
        
        try:
            s2_check = spell_checker.check(s2).checked
        except:
            s2_check = s2
            print("ERROR: " + s2)

        newdata['sentence_1'][i] = s1_check
        newdata['sentence_2'][i] = s2_check
    
    return newdata


def clean_string(text):
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl='', string=text)

    pattern = '[^\w\s\n]'         # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)

    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', string=text)

    return text

def clean(data):
    newdata = data.copy()

    newdata['sentence_1'] = newdata['sentence_1'].apply(clean_string)
    newdata['sentence_2'] = newdata['sentence_2'].apply(clean_string)

    print(newdata['sentence_1'][:5])
    return newdata


def normalize(data):
    newdata = data.copy()

    newdata['sentence_1'].apply(repeat_normalize)
    newdata['sentence_2'].apply(repeat_normalize)

    return newdata



### AUGMENTATION

def base_augmentation(data):
    print('='*10)
    print('base augmentation')
    print('='*10)
    return data

def switch(data):
    newdata = data.copy()

    newdata['sentence_1'] = data['sentence_2']
    newdata['sentence_2'] = data['sentence_1']

    print(len(newdata))

    return pd.concat([data, newdata])

def switch_v2(data):
    newdata = data.copy()[data['label']!=0.0]
    
    newdata['sentence_1'] = data['sentence_2']
    newdata['sentence_2'] = data['sentence_1']

    print(len(newdata))

    return pd.concat([data, newdata])

# def eda_x2(data):
#     newdata = data.copy()

#     for i, (s1, s2) in enumerate(zip(data['sentence_1'], data['sentence_2'])):
#         s1_aug = EDA(s1, alpha_sr=.0, alpha_ri=.0, alpha_rs=0.1, p_rd=0.1, num_aug=1)[0]
#         s2_aug = EDA(s2, alpha_sr=.0, alpha_ri=.0, alpha_rs=0.1, p_rd=0.1, num_aug=1)[0]
        
#         newdata['sentence_1'][i] = s1_aug
#         newdata['sentence_2'][i] = s2_aug

#         print(s1_aug, s2_aug)

#     return pd.concat([data, newdata])




### UTILITIES

def scatterplot(model, trainer, datamodule, filename):
    ##plot

    pred_path = datamodule.predict_path
    datamodule.predict_path = datamodule.dev_path

    true = torch.tensor(pd.read_csv(datamodule.dev_path)['label'])
    pred = torch.cat(trainer.predict(model=model, datamodule=datamodule))

    data = pd.DataFrame({
        'true': pd.Series(true),
        'pred': pd.Series(pred)
    })

    plot = sns.scatterplot(data, x='true', y='pred')
    fig = plot.get_figure()
    fig.savefig('./save/plot/' + filename) 

    datamodule.predict_path = pred_path

    return 0