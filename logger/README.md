# Overview
log 즉, 학습의 결과물들을 저장하는 모듈

# 구성
- logger.py : epoch등의 단위당 loss 또는 metric의 결과를 저장하는 곳
    - handler에서 지정한 단위로 handler['filename'] = str(save_dir / handler['filename'])
    이처럼 값을 저장함
- logger_config : json파일로 원하는 로그 형태를 입력할 수 있는 듯 하다.. 
    - handler의 DEBUG level등 잘 모르는 용어가 많아서 아시는 분 알려주세요.