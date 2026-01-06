# V2 Ensemble Auto-Labeling System

**Version**: 2.0
**Architecture**: 3-Model Ensemble (YOLO, Model2, Model3)
**API Base**: `/api/v2`

---

## Overview

V2 Ensemble Auto-Labeling System은 3개의 독립적인 객체 탐지 모델을 앙상블하여 자동 라벨링 품질을 향상시키는 시스템입니다.

### Key Features

- ✅ **3-Model Ensemble**: YOLO, Model2, Model3 독립 학습 및 추론
- ✅ **Model-Specific Data Management**: 각 모델별 GT 및 Unlabeled 데이터 독립 관리
- ✅ **Ensemble Classification**: PASS_THREE, PASS_TWO, FAIL, MISS 4단계 분류
- ✅ **Round Repetition**: FAIL 데이터 재학습 → 전체 Unlabeled 재추론 (Round 0, 1, 2)
- ✅ **Worker Proxy Architecture**: API Server ↔ Worker Server 분리
- ✅ **File-Based Event Logging**: JSON 파일 기반 이벤트 영구 저장
- ✅ **Export System**: Round별 및 최종 결과 ZIP 생성/다운로드
- ✅ **Spring Boot Polling Support**: 프론트엔드 폴링에 최적화된 Event API

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Spring Boot Frontend                         │
│  - Loop 시작/상태 조회                                       │
│  - 이벤트 폴링 (2초마다)                                      │
│  - 결과 Export 다운로드                                      │
└─────────────────────┬───────────────────────────────────────┘
                      │ REST API
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          FastAPI API Server (v1-api:8010)                   │
│  - Data Ingest (GT/Unlabeled 업로드)                        │
│  - Loop Proxy (Worker로 요청 전달)                          │
│  - Event Storage (Worker 콜백 수신)                         │
│  - Export Management (ZIP 생성/서빙)                        │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP Proxy
                      ▼
┌─────────────────────────────────────────────────────────────┐
│        FastAPI Worker Server (v1-worker:8011)               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Ensemble Loop Worker (별도 Thread)                    │ │
│  │  1. 3개 모델 병렬 추론 (YOLO, Model2, Model3)         │ │
│  │  2. 앙상블 분류 (PASS_THREE/TWO/FAIL/MISS)            │ │
│  │  3. Bbox 병합 (NMS + 평균화)                          │ │
│  │  4. Round 반복 (FAIL 재학습 → 전체 재추론)            │ │
│  │  5. 이벤트 콜백 (API Server로 진행 상황 전송)         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ AI Model Layer                                         │ │
│  │  - Model 1: YOLO (ultralytics)                         │ │
│  │  - Model 2: Custom COCO Model                          │ │
│  │  - Model 3: Custom COCO Model                          │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              File Storage (Local/NAS)                        │
│  auto_labeling/v_1/data/                                     │
│  ├── gt_data/           (모델별 GT 데이터)                  │
│  ├── unlabeled/         (모델별 Unlabeled 이미지)           │
│  ├── results/           (Loop 결과)                         │
│  ├── exports/           (Export 결과 캐시)                  │
│  └── logs/events/       (이벤트 로그 JSON)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. 준비 (GT + Unlabeled 업로드)

각 모델별로 GT 및 Unlabeled 데이터를 업로드합니다.

```bash
# YOLO GT 업로드 및 등록
curl -X POST http://localhost:8010/api/v2/yolo/gt/upload \
  -F "file=@gt_yolo.zip" \
  -F "sourceName=client_A"

curl -X POST "http://localhost:8010/api/v2/yolo/gt/register?ingestId=gt_yolo_xxx"

# YOLO Unlabeled 업로드
curl -X POST http://localhost:8010/api/v2/yolo/unlabeled/upload \
  -F "file=@unlabeled_yolo.zip"

# Model2, Model3도 동일하게 반복
```

### 2. Loop 실행

```bash
# Loop 시작
curl -X POST http://localhost:8010/api/v2/loop/start \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["yolo", "model2", "model3"],
    "configOverride": {
      "maxRounds": 3,
      "confThreshold": 0.5
    }
  }'

# Response:
# {
#   "loopId": "loop_abc123",
#   "runId": "run_20250106_120000_xyz789",
#   "status": "STARTED"
# }
```

### 3. 진행 상황 모니터링 (이벤트 폴링)

```bash
# 최신 이벤트 조회 (Spring Boot가 2초마다 폴링)
curl "http://localhost:8010/api/v2/events/latest?runId=run_20250106_120000_xyz789"

# Loop 상태 조회
curl "http://localhost:8010/api/v2/loop/status/loop_abc123"
```

### 4. 결과 Export

```bash
# Round 0 Export
curl -X POST "http://localhost:8010/api/v2/export/round?loopId=loop_abc123&runNumber=0"

# Round 0 다운로드
curl -O "http://localhost:8010/api/v2/export/round/download?loopId=loop_abc123&runNumber=0"

# Final Export
curl -X POST "http://localhost:8010/api/v2/export/final?loopId=loop_abc123"

# Final 다운로드
curl -O "http://localhost:8010/api/v2/export/final/download?loopId=loop_abc123"
```

---

## API Documentation

자세한 API 문서는 다음 파일을 참조하세요:

- **[API_REFERENCE.md](./API_REFERENCE.md)**: 전체 API 상세 문서 (Swagger 스타일)
- **[API_ENDPOINTS.md](./API_ENDPOINTS.md)**: API 엔드포인트 빠른 참조 및 예제

### API 요약

| Category | Endpoints | Description |
|----------|-----------|-------------|
| **Data Ingest** | 18 | GT/Unlabeled 업로드 및 관리 (모델당 6개 × 3) |
| **Loop** | 2 | Loop 시작 및 상태 조회 |
| **Event** | 4 | 이벤트 수신 및 조회 |
| **Export** | 4 | Round별 및 최종 결과 Export |
| **Total** | **28** | |

---

## Ensemble Classification

각 이미지에 대해 3개 모델의 PASS/FAIL 결과를 조합하여 4단계로 분류합니다.

| YOLO | Model2 | Model3 | Category | Description |
|------|--------|--------|----------|-------------|
| ✅ PASS | ✅ PASS | ✅ PASS | **PASS_THREE** | 3개 모두 PASS (최고 품질) |
| ✅ PASS | ✅ PASS | ❌ FAIL | **PASS_TWO** | 2개 PASS, 1개 FAIL (중간 품질) |
| ✅ PASS | ❌ FAIL | ✅ PASS | **PASS_TWO** | 2개 PASS, 1개 FAIL |
| ❌ FAIL | ✅ PASS | ✅ PASS | **PASS_TWO** | 2개 PASS, 1개 FAIL |
| ✅ PASS | ❌ FAIL | ❌ FAIL | **FAIL** | 1개 PASS, 2개 FAIL (재학습 대상) |
| ❌ FAIL | ✅ PASS | ❌ FAIL | **FAIL** | 1개 PASS, 2개 FAIL |
| ❌ FAIL | ❌ FAIL | ✅ PASS | **FAIL** | 1개 PASS, 2개 FAIL |
| ❌ FAIL | ❌ FAIL | ❌ FAIL | **MISS** | 3개 모두 FAIL (라벨 없음) |

**PASS/FAIL 기준**:
- PASS: Confidence ≥ threshold (default: 0.5)
- FAIL: Confidence < threshold

---

## Round Repetition Strategy

```
Round 0:
  Input:    모든 Unlabeled 이미지 (예: 1000장)
  Process:  3모델 병렬 추론 → 앙상블 분류
  Output:   PASS_THREE (650), PASS_TWO (200), FAIL (100), MISS (50)

Round 1:
  Input:    전체 Unlabeled 이미지 (1000장)
  Retrain:  Round 0의 FAIL (100장)로 각 모델 재학습
  Process:  재학습된 모델로 전체 Unlabeled 재추론
  Output:   PASS_THREE (720), PASS_TWO (180), FAIL (70), MISS (30)

Round 2:
  Input:    전체 Unlabeled 이미지 (1000장)
  Retrain:  Round 1의 FAIL (70장)로 각 모델 재학습
  Process:  재학습된 모델로 전체 Unlabeled 재추론
  Output:   PASS_THREE (750), PASS_TWO (170), FAIL (60), MISS (20)
```

**Current**: 3 Rounds 고정 (0, 1, 2)

**Future**: Fail ratio 기반 동적 종료 조건
- `failMissRatio < failThreshold` 시 조기 종료
- `patience` 기반 early stopping

---

## Data Directory Structure

```
auto_labeling/v_1/data/
├── gt_data/                    # 모델별 GT 데이터
│   ├── yolo/
│   │   ├── GT_gt_yolo_20250106_120000_abc123/
│   │   │   ├── images/
│   │   │   ├── labels/
│   │   │   └── data.yaml
│   │   └── GT.file → GT_gt_yolo_20250106_120000_abc123  (symlink)
│   ├── model2/  (동일 구조)
│   └── model3/  (동일 구조)
│
├── unlabeled/                  # 모델별 Unlabeled 이미지
│   ├── yolo/images/
│   ├── model2/images/
│   └── model3/images/
│
├── results/                    # Loop 결과
│   └── loop_{loop_id}/
│       ├── run_0/              (Round 0 결과)
│       │   ├── PASS_THREE/
│       │   │   ├── images/
│       │   │   └── labels/
│       │   ├── PASS_TWO/
│       │   ├── FAIL/
│       │   └── MISS/
│       ├── run_1/              (Round 1 결과)
│       └── run_2/              (Round 2 결과)
│
├── exports/                    # Export 결과 캐시
│   └── {loop_id}/
│       ├── run_0.zip
│       ├── run_1.zip
│       ├── run_2.zip
│       └── final.zip
│
├── raw_ingest/                 # 업로드 임시 저장
│
└── logs/
    └── events/                 # 이벤트 로그 JSON
        └── {runId}/
            ├── 20250106_120000_123456_LOOP_STARTED.json
            ├── 20250106_120001_234567_ROUND_RESULT.json
            └── 20250106_120030_345678_LOOP_DONE.json
```

---

## Label Format

### YOLO Format (Normalized)

```
# Format: class cx cy w h
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.15 0.25

# 좌표는 0.0 ~ 1.0 정규화
# cx, cy: 중심 좌표
# w, h: 너비, 높이
```

### GT ZIP Structure

```
gt.zip
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── labels/
    ├── img001.txt  (YOLO format)
    ├── img002.txt
    └── ...
```

---

## Event System

### Event Flow

```
Worker → POST /api/v2/events → API Server → JSON 파일 저장
                                              ↓
                                    logs/events/{runId}/
                                              ↓
Spring Boot ← GET /api/v2/events/latest ← API Server
(2초마다 폴링)
```

### Event Types

- `LOOP_STARTED`: Loop 시작
- `LOOP_DONE`: Loop 완료
- `LOOP_FAILED`: Loop 실패
- `ROUND_RESULT`: Round 결과
- `EXPORT_FINAL_READY`: Final export 준비
- `EXPORT_FINAL_DONE`: Final export 완료
- `EXPORT_FINAL_FAILED`: Final export 실패

### Event Storage

```json
// logs/events/run_20250106_120000_xyz789/20250106_120000_123456_LOOP_STARTED.json
{
  "receivedAt": "2025-01-06T12:00:00Z",
  "event": {
    "eventType": "LOOP_STARTED",
    "runId": "run_20250106_120000_xyz789",
    "jobId": "job_20250106_120000",
    "message": "Loop started",
    "payload": {
      "loopId": "loop_abc123",
      "models": ["yolo", "model2", "model3"]
    }
  }
}
```

---

## Export System

### Round Export

**ZIP Structure**:
```
run_0.zip
├── PASS_THREE/
│   ├── images/
│   └── labels/
├── PASS_TWO/
│   ├── images/
│   └── labels/
├── FAIL/
│   ├── images/
│   └── labels/
└── MISS/
    └── images/
```

### Final Export

**ZIP Structure**:
```
final.zip
├── PASS/         (모든 Round의 PASS_THREE + PASS_TWO 통합)
│   ├── images/
│   └── labels/
├── FAIL/         (최종 Round의 FAIL)
│   ├── images/
│   └── labels/
└── MISS/         (최종 Round의 MISS)
    └── images/
```

**Processing**:
1. 모든 Round의 `PASS_THREE`, `PASS_TWO` → `PASS` 병합
2. 최종 Round의 `FAIL` → `FAIL` 복사
3. 최종 Round의 `MISS` → `MISS` 복사

---

## Configuration

### Environment Variables

```bash
# Worker Server URL
WORKER_BASE_URL=http://v1-worker:8011

# Event Storage
V2_EVENTS_ROOT=/mnt/nas/v2_events

# Data Root
V2_DATA_ROOT=/mnt/nas/v2_data
```

### Loop Configuration

```json
{
  "maxRounds": 3,           // 최대 Round 수
  "confThreshold": 0.5,     // PASS/FAIL 판정 threshold
  "failThreshold": 0.01,    // Fail ratio 종료 조건
  "minFailCount": 100,      // 최소 FAIL 개수 (재학습 조건)
  "patience": 2,            // Early stopping patience
  "iouThreshold": 0.5       // NMS IoU threshold
}
```

---

## Error Handling

### Standard Error Response

```json
{
  "detail": "Error message",
  "status_code": 400
}
```

### Common Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 404 | Not Found |
| 500 | Internal Server Error |
| 503 | Service Unavailable (Worker unreachable) |

---

## Development

### Project Structure

```
auto_labeling/v_2/
├── api/
│   ├── dto/                # DTOs (Pydantic models)
│   │   ├── base.py
│   │   ├── data_ingest.py
│   │   ├── loop.py
│   │   ├── event.py
│   │   ├── export.py
│   │   └── results.py
│   └── routers/            # API Routers
│       ├── data_ingest.py
│       ├── loop.py
│       ├── events.py
│       └── export_v2.py
├── services/               # Business Logic
│   └── data_manager.py
├── docs/                   # Documentation
│   ├── README.md
│   ├── API_REFERENCE.md
│   └── API_ENDPOINTS.md
└── DATA_STRUCTURE.md       # Data structure documentation
```

### Dependencies

```bash
# Python 3.9+
fastapi
uvicorn
pydantic
httpx
python-multipart
PyYAML
ultralytics  # YOLO
```

---

## Testing

### Manual Testing

```bash
# 1. API Server 시작
cd auto_labeling/v_1
uvicorn api.server:app --host 0.0.0.0 --port 8010

# 2. Worker Server 시작 (별도 터미널)
uvicorn worker.server:app --host 0.0.0.0 --port 8011

# 3. GT 업로드 테스트
curl -X POST http://localhost:8010/api/v2/yolo/gt/upload \
  -F "file=@test_gt.zip"

# 4. Loop 시작 테스트
curl -X POST http://localhost:8010/api/v2/loop/start \
  -H "Content-Type: application/json" \
  -d '{"models": ["yolo", "model2", "model3"]}'
```

---

## Migration from V1

### Key Differences

| Feature | V1 (Student-Teacher) | V2 (Ensemble) |
|---------|---------------------|---------------|
| Models | 2 (Student, Teacher) | 3 (YOLO, Model2, Model3) |
| Architecture | Single GT/Unlabeled | Model-specific GT/Unlabeled |
| Classification | PASS/FAIL | PASS_THREE/TWO/FAIL/MISS |
| Round Strategy | FAIL only re-inference | FAIL retrain → All re-inference |
| API Prefix | `/api/v1` | `/api/v2` |

### Migration Steps

1. **데이터 준비**: 각 모델별 GT 및 Unlabeled 데이터 분리
2. **API 변경**: `/api/v1` → `/api/v2` 엔드포인트 변경
3. **Response 구조**: 4단계 분류 대응 (PASS_THREE, PASS_TWO, FAIL, MISS)
4. **Event Polling**: 동일한 폴링 패턴 유지 가능

---

## FAQ

**Q: GT는 모델별로 다른 데이터를 사용해야 하나요?**
A: 네, 각 모델이 독립적으로 학습하므로 모델별로 최적화된 GT를 사용할 수 있습니다. 동일한 GT를 사용해도 무방합니다.

**Q: Round는 몇 번까지 실행되나요?**
A: 현재는 3번 (Round 0, 1, 2) 고정입니다. 추후 Fail ratio 기반 동적 종료 조건이 추가될 예정입니다.

**Q: PASS_TWO는 어떻게 활용하나요?**
A: PASS_TWO는 2개 모델이 PASS한 중간 품질 데이터입니다. 최종 Export 시 PASS로 통합되어 활용됩니다.

**Q: Worker Server가 다운되면 어떻게 되나요?**
A: API Server가 503 에러를 반환하며, 이벤트는 JSON 파일로 영구 저장되므로 재시작 후 복구 가능합니다.

**Q: Export ZIP은 어디에 저장되나요?**
A: `data/exports/{loop_id}/` 디렉토리에 캐시되며, 재생성 방지를 위해 기존 파일을 덮어씁니다.

---

## Support

- **Issues**: GitHub Issues
- **Documentation**: [API_REFERENCE.md](./API_REFERENCE.md)
- **Contact**: V2 Ensemble Team

---

**Version**: 2.0
**Last Updated**: 2025-01-06
