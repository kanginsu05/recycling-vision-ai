# 데이터 준비

이미지 데이터는 저장소에 포함하지 않습니다. 아래 구조로 `data/` 폴더에 넣으면 모든 스크립트가 그대로 동작합니다.

```
data/
├── train/
│   ├── cardboard/  *.jpg
│   ├── glass/      *.jpg
│   ├── metal/      *.jpg
│   ├── paper/      *.jpg
│   ├── plastic/    *.jpg
│   └── trash/      *.jpg
├── test/           *.jpg          # 정답이 없는 캐글 테스트 이미지 (선택)
└── sample_submission.csv          # (선택)
```

## 데이터 출처

- 대회: **2026 청년 AI 머신러닝/딥러닝 – 분리수거 AI 챌린지** (교육과정 내부 Kaggle 대회 `2026-ai-34`)
- 원본 데이터셋: **TrashNet** — Gary Thung & Mindy Yang, *Classification of Trash for Recyclability Status* (Stanford CS229, 2016), <https://github.com/garythung/trashnet>
  - 6개 클래스, 512×384 RGB, 흰 배경에서 촬영 · MIT License
  - README 그림과 데모 예시 이미지(`assets/`)에 TrashNet 이미지 일부가 포함되어 있습니다.
  - 대회용으로 학습 2,151장 / 테스트 373장으로 나뉘어 제공되었습니다.

대회 데이터에 접근할 수 없다면 TrashNet 원본(`dataset-resized.zip`)을 내려받아 클래스별 폴더를 `data/train/` 아래에 두면 됩니다. 이 경우 이미지 수가 조금 달라서 결과 수치도 약간 달라질 수 있습니다.

처음 실행할 때 모든 이미지를 256×256으로 한 번 디코딩해 `data/cache/`에 저장합니다(약 500MB).
