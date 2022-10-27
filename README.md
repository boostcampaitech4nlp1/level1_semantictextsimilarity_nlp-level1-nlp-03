# New Baseline

* 수정한 newbase입니다. 
* 한성님이 주신 베이스 코드에 수정을 했습니다. 

### 추가한 것
* 시드 고정, force_download=True => 재현가능
* utils에 base_preprocess 함수 추가
* shellscript.sh 추가 -> repo에서 sh hellscript.sh 실행하면 실행가능

### 그 외
* util과 model.py에 각각 함수와 클래스를 추가한 후 터미널에서 --preprocess='base_preprocess' --model_arch='BaseModel' 같이 이름으로 불러올 수 있습니다. 
* 기존에 torch.save는 모델 내의 하이퍼파라미터만 저장을 하기 때문에 저희가 무슨 preprocess 함수를 썼는지, 무슨 모델구조를 썼는지는 저장이 되지 않습니다. 따로 저장을 해야할 것 같습니다. (util에 save_config함수 참조)