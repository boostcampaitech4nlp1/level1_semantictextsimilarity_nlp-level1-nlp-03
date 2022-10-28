import pandas as pd
from hanspell import spell_checker

data_df = pd.read_csv('/opt/ml/data/train.csv')
trian_df = data_df['sentence_1']

for s in trian_df:
    before = s
    after = spell_checker.check(s)
    after = after.as_dict()
    if before != after['checked']:
        print(after['original'])
        print(after['checked'])
        



