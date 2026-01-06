# V2 데이터 구조 문서

## 전체 디렉토리 구조

```
auto_labeling/v_2/data/
├── gt_data/                      # 모델별 GT 데이터
│   ├── yolo/
│   │   ├── GT_202601/
│   │   │   ├── images/
│   │   │   ├── labels/
│   │   │   └── data.yaml
│   │   └── GT.file → GT_202601  # 현재 활성 GT (심볼릭 링크)
│   ├── model2/
│   │   └── (동일 구조)
│   └── model3/
│       └── (동일 구조)
│
├── unlabeled/                    # 모델별 Unlabeled 이미지
│   ├── yolo/
│   │   └── images/
│   ├── model2/
│   │   └── images/
│   └── model3/
│       └── images/
│
├── results/                      # Loop 결과
│   └── loop_{loop_id}/           # Loop 단위
│       ├── run_0/                # Round 0 결과
│       │   ├── PASS_THREE/
│       │   │   ├── images/
│       │   │   └── labels/       # 3모델 bbox 평균화
│       │   ├── PASS_TWO/
│       │   │   ├── images/
│       │   │   └── labels/
│       │   ├── FAIL/
│       │   │   ├── images/
│       │   │   └── labels/
│       │   └── MISS/
│       │       └── images/       # 라벨 없음
│       ├── run_1/                # Round 1 결과
│       │   └── (동일 구조)
│       ├── run_2/                # Round 2 결과
│       │   └── (동일 구조)
│       └── final/                # 최종 병합 결과
│           ├── PASS/             # 모든 Round PASS 통합
│           │   ├── images/
│           │   └── labels/
│           ├── FAIL/             # 최종 Round FAIL
│           │   ├── images/
│           │   └── labels/
│           └── MISS/             # 최종 Round MISS
│               └── images/
│
├── exports/                      # Export 결과 캐시
│   └── {loop_id}/
│       ├── run_0.zip             # Round 0 ZIP
│       ├── run_1.zip             # Round 1 ZIP
│       ├── run_2.zip             # Round 2 ZIP
│       └── final.zip             # 최종 결과 ZIP
│
├── raw_ingest/                   # 업로드 임시 저장
│   ├── gt_{model}_{timestamp}/
│   └── unlabeled_{model}_{timestamp}/
│
└── logs/                         # 로그
    ├── events/{loop_id}/         # 이벤트 로그 JSON
    │   └── {timestamp}_*.json
    └── previews/{loop_id}/       # 프리뷰 이미지
        ├── run_0/
        ├── run_1/
        └── run_2/
```

## Round별 데이터 흐름

### Round 0 (사전 학습 모델)
```
Input: 모든 Unlabeled 이미지 (1000장)
Process: 3모델 병렬 추론 → 앙상블 분류
Output:
  - PASS_THREE: 650장 (3개 모두 PASS)
  - PASS_TWO: 200장 (2개 PASS, 1개 FAIL)
  - FAIL: 100장 (1개 PASS, 2개 FAIL)
  - MISS: 50장 (3개 모두 FAIL)
```

### Round 1 (FAIL로 재학습 후)
```
Retrain: Round 0의 FAIL 100장 + GT Anchor로 3개 모델 재학습
Input: 모든 Unlabeled 이미지 (1000장 전체)
Process: 재학습된 모델로 전체 재추론 → 앙상블 분류
Output:
  - PASS_THREE: 700장 (개선)
  - PASS_TWO: 220장
  - FAIL: 60장 (감소)
  - MISS: 20장 (감소)
```

### Round 2 (FAIL로 재학습 후)
```
Retrain: Round 1의 FAIL 60장 + GT Anchor로 3개 모델 재학습
Input: 모든 Unlabeled 이미지 (1000장 전체)
Process: 재학습된 모델로 전체 재추론 → 앙상블 분류
Output:
  - PASS_THREE: 750장
  - PASS_TWO: 210장
  - FAIL: 30장
  - MISS: 10장
종료: 현재는 Round 2에서 종료 (고정)
```

## 파일명 규칙

### GT 데이터
- 디렉토리: `GT_{timestamp}` (예: GT_20250106_120000)
- 심볼릭 링크: `GT.file` → 현재 활성 GT

### Unlabeled 데이터
- 디렉토리: `unlabeled/{model}/images/`
- 파일명: 원본 파일명 유지

### 결과 데이터
- Loop ID: `loop_{uuid}` (예: loop_abc123)
- Run 디렉토리: `run_{round_number}` (예: run_0, run_1, run_2)

### Export 파일
- Round별: `{loop_id}/run_{round}.zip`
- Final: `{loop_id}/final.zip`

### 이벤트 로그
- 파일명: `{timestamp}_{sequence}_{event_type}.json`
- 예: `20250106_120000_000001_LOOP_STARTED.json`

## 라벨 형식

### YOLO 형식 (normalized)
```
{class_id} {cx} {cy} {w} {h}
0 0.5 0.5 0.3 0.3
```

### COCO 형식 (absolute)
```json
{
  "image_id": 1,
  "category_id": 0,
  "bbox": [x1, y1, width, height],
  "score": 0.95
}
```

## 주의사항

1. **모든 Round에서 전체 Unlabeled 재추론**
   - Round 1+는 FAIL만 재추론하는 것이 아님
   - 재학습된 모델로 전체 이미지를 처음부터 다시 추론

2. **Round 수 고정**
   - 현재: Round 0, 1, 2 (총 3회) 고정
   - 향후: 동적 종료 조건 추가 가능

3. **모델별 독립 GT/Unlabeled**
   - 각 모델은 독립적인 GT와 Unlabeled 데이터 보유
   - 앙상블 시 3개 모델의 결과만 통합

4. **심볼릭 링크 사용**
   - GT.file은 항상 최신 GT 버전을 가리킴
   - 심볼릭 링크 실패 시 복사 모드로 대체 가능
