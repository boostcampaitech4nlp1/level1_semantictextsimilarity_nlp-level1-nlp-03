# Overview
- 김한성_T4057
- Sentence-BERT 구현 템플릿입니다.
- Sentence-BERT는 임베딩 벡터를 학습하는 것입니다.
- 이후 fine tuning한 모델로 학습을진행해 보세요!


```python3
# -> 임베딩 벡터 학습
python3 sbert_train.py --config base_config 
# -> 학습된 임베딩 벡터로 fine-tune
python3 train.py --config base_config 
# -> 최종모델로 추론
python3 inference.py --config test_config 
 ```
