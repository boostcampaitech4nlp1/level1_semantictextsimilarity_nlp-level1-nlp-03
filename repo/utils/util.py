import seaborn as sns
import pandas as pd

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
    print(data[:10])
    return data


def scatterplot(true, pred, filename):
    ##plot
    data = pd.DataFrame({
        'true': pd.Series(true),
        'pred': pd.Series(pred)
    })

    plot = sns.scatterplot(data, x='true', y='pred')
    fig = plot.get_figure()
    fig.savefig('./save/' + filename) 

    return 0


def save_config(args, filename):
    # 모델 자체의 파라미터는 자동으로 저장되지만 
    # 모델 외의 전처리, 커스텀한 모델 구조 등은 저장이 안되니까
    # 그런거는 자동으로 저장되게 하는게 좋을듯 해요
    # ex) model_arch = TransAndCNN
    # ex) preprocess = withoutEnglish
    # 아마 json?

    return 0