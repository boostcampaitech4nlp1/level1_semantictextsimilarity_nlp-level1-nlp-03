import re

def remove_word(sentences):
    

    for i,sentence in enumerate(sentences):
        punct = sentence[-1]
        # 자모 제거
        sentence = re.sub("[ㄱ-ㅎ]+", ' ', sentence.strip())
        # 특수문자 제거
        sentence = re.sub("[^가-힣a-zA-Z0-9]+", ' ', sentence.strip())
        # 중복공백 제거
        sentence = re.sub(" +", ' ', sentence.strip())
        # 구두점
        if punct in ['!', '.', '?', '~']:
            sentence += punct

        sentences[i] = sentence
    return sentences