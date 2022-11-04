from hanspell import spell_checker
from sentence_transformers.readers import InputExample

def data_re(DF):
    def no_space(sent):
        sent = sent.replace('&',',').lower()
        spelled_sent = spell_checker.check(sent)
        hanspell_sent = spelled_sent.checked
        return hanspell_sent

    DF['sentence_1'] = DF['sentence_1'].apply(no_space)
    DF['sentence_2'] = DF['sentence_2'].apply(no_space)
    return DF

def make_sts_input_example(dataset):
    input_examples = []
    for i, data in enumerate(dataset):
        sentence1 = data['sentence_1']
        sentence2 = data['sentence_2']
        try:
            score = (data['label']) / 5.0  # normalize 0 to 1
        except:
            score = None
        input_examples.append(InputExample(texts=[sentence1, sentence2], label=score))

    return input_examples