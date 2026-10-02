# legacy — 대회 당시 원본 코드

`garbage_classification_3.py`는 대회 기간에 최종 제출(Public 0.75000, 6위)에 사용한 스크립트를 **수정 없이** 보관한 것입니다.

- 직접 설계한 4블록 CNN(DeepCNN) · 96×96 입력
- 5-Fold 교차검증 앙상블 + 좌우반전 TTA
- 클래스 가중치 + Cosine LR 스케줄러

대회 이후 코드를 모듈로 나누고(`src/`), 모든 단계를 같은 Hold-out 데이터로 다시 평가하도록 재구성한 버전이 `experiments/01_baselines.py`입니다.
