import pandas as pd

def softvoting(output1, output2):
    result = output1.copy()

    for i, (o1, o2) in enumerate(zip(output1['target'], output2['target'])):
        result['target'][i] = round((o1 + o2) / 2, 1)
    
    return result

if __name__ == "__main__":
    output1 = pd.read_csv('./save/output/output_9202.csv')
    output2 = pd.read_csv('./save/output/output_9126.csv')

    result = softvoting(output1, output2)

    result.to_csv('./save/output/sv_9202_9126.csv', index=False)
