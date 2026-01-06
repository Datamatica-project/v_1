# V2 Ensemble Auto-Labeling API Reference

**Version**: 2.0
**Base URL**: `/api/v2`
**Architecture**: 3-Model Ensemble (YOLO, Model2, Model3)

---

## Table of Contents

1. [Data Ingest APIs](#1-data-ingest-apis)
   - [GT Upload & Registration](#gt-upload--registration)
   - [Unlabeled Upload](#unlabeled-upload)
2. [Loop APIs](#2-loop-apis)
3. [Event APIs](#3-event-apis)
4. [Export APIs](#4-export-apis)
5. [Data Structures](#5-data-structures)

---

## 1. Data Ingest APIs

각 모델(YOLO, Model2, Model3)별로 독립적인 GT 및 Unlabeled 데이터 관리를 제공합니다.

### GT Upload & Registration

#### 1.1 Upload GT ZIP (YOLO)

**Endpoint**: `POST /api/v2/yolo/gt/upload`

**Description**: YOLO 모델용 GT(Ground Truth) 데이터 ZIP 파일 업로드

**Request**:
```http
POST /api/v2/yolo/gt/upload
Content-Type: multipart/form-data

Parameters:
- file: ZIP file (required)
  - Must contain: images/, labels/
  - Labels in YOLO format: class cx cy w h
- sourceName: string (optional) - GT 제공처
- datasetName: string (optional) - 데이터셋 이름
```

**Response**:
```json
{
  "ingestId": "gt_yolo_20250106_120000_abc123",
  "modelName": "yolo",
  "status": "UPLOADED",
  "extractedPath": "/workspace/.../raw_ingest/gt_yolo_xxx/extracted",
  "next": "/api/v2/yolo/gt/register?ingestId=gt_yolo_20250106_120000_abc123"
}
```

**ZIP Structure**:
```
gt.zip
├── images/
│   ├── img001.jpg
│   └── img002.jpg
└── labels/
    ├── img001.txt  (YOLO format: class cx cy w h)
    └── img002.txt
```

---

#### 1.2 Register GT (YOLO)

**Endpoint**: `POST /api/v2/yolo/gt/register`

**Description**: 업로드된 GT를 검증하고 현재 활성 GT로 등록

**Request**:
```http
POST /api/v2/yolo/gt/register?ingestId=gt_yolo_20250106_120000_abc123&copyMode=symlink&strict=false

Query Parameters:
- ingestId: string (required) - Upload에서 받은 ingest ID
- copyMode: "symlink" | "copy" (default: "symlink")
- strict: boolean (default: false) - 라벨 검증 실패 시 중단 여부
```

**Response**:
```json
{
  "ingestId": "gt_yolo_20250106_120000_abc123",
  "modelName": "yolo",
  "status": "DONE",
  "registeredPath": ".../data/gt_data/yolo/GT_gt_yolo_xxx",
  "currentGTPath": ".../data/gt_data/yolo/GT.file",
  "summary": {
    "ok": 1500,
    "skip": 0,
    "error": 0
  }
}
```

**Processing**:
1. 라벨 검증 (YOLO format: `class cx cy w h`)
2. GT 버전 디렉토리 생성: `gt_data/yolo/GT_{ingestId}/`
3. 현재 GT 심볼릭 링크 갱신: `gt_data/yolo/GT.file → GT_{ingestId}`
4. `data.yaml` 생성

---

#### 1.3 List GT Versions (YOLO)

**Endpoint**: `GET /api/v2/yolo/gt/versions`

**Description**: YOLO 모델의 GT 버전 목록 조회

**Response**:
```json
{
  "modelName": "yolo",
  "versions": [
    {
      "versionId": "GT_gt_yolo_20250106_120000_abc123",
      "modelName": "yolo",
      "isCurrent": true,
      "createdAt": "2025-01-06T12:00:00+09:00",
      "imageCount": 1500,
      "sourceName": "client_A",
      "datasetName": "202501_batch1"
    }
  ]
}
```

---

### Unlabeled Upload

#### 1.4 Upload Unlabeled Images (YOLO)

**Endpoint**: `POST /api/v2/yolo/unlabeled/upload`

**Description**: YOLO 모델용 Unlabeled 이미지 ZIP 업로드

**Request**:
```http
POST /api/v2/yolo/unlabeled/upload
Content-Type: multipart/form-data

Parameters:
- file: ZIP file (required) - 이미지 파일들 (.jpg, .jpeg, .png, .bmp, .webp)
- datasetName: string (optional)
```

**Response**:
```json
{
  "ingestId": "unlabeled_yolo_20250106_120000_xyz789",
  "modelName": "yolo",
  "status": "DONE",
  "addedImages": 500,
  "unlabeledDir": ".../data/unlabeled/yolo/images"
}
```

**Processing**:
- ZIP 내 모든 이미지 파일을 `data/unlabeled/yolo/images/`로 복사
- 중복 파일명 자동 처리 (suffix 추가)

---

#### 1.5 Get Unlabeled Info (YOLO)

**Endpoint**: `GET /api/v2/yolo/unlabeled/info`

**Description**: YOLO Unlabeled 이미지 정보 조회

**Response**:
```json
{
  "modelName": "yolo",
  "imageCount": 500,
  "unlabeledDir": ".../data/unlabeled/yolo/images"
}
```

---

### Model2 & Model3 APIs

동일한 API 구조가 Model2, Model3에 대해서도 제공됩니다:

**Model2 Endpoints**:
- `POST /api/v2/model2/gt/upload`
- `POST /api/v2/model2/gt/register`
- `GET /api/v2/model2/gt/versions`
- `POST /api/v2/model2/unlabeled/upload`
- `GET /api/v2/model2/unlabeled/info`

**Model3 Endpoints**:
- `POST /api/v2/model3/gt/upload`
- `POST /api/v2/model3/gt/register`
- `GET /api/v2/model3/gt/versions`
- `POST /api/v2/model3/unlabeled/upload`
- `GET /api/v2/model3/unlabeled/info`

**Total**: 18 endpoints (6 per model × 3 models)

---

## 2. Loop APIs

Worker Server로 요청을 프록시하여 앙상블 Loop 실행을 제어합니다.

### 2.1 Start Ensemble Loop

**Endpoint**: `POST /api/v2/loop/start`

**Description**: 3모델 앙상블 Loop 시작 (Worker로 프록시)

**Request**:
```json
{
  "models": ["yolo", "model2", "model3"],
  "configOverride": {
    "maxRounds": 3,
    "confThreshold": 0.5,
    "failThreshold": 0.01,
    "minFailCount": 100,
    "patience": 2,
    "iouThreshold": 0.5
  }
}
```

**Response**:
```json
{
  "loopId": "loop_abc123",
  "runId": "run_20250106_120000_xyz789",
  "status": "STARTED",
  "message": "Ensemble loop started"
}
```

**Processing Flow**:
1. API Server → Worker Server 프록시 (`http://v1-worker:8011/api/v2/loop/start`)
2. Worker가 별도 Thread에서 Ensemble Loop 실행 시작
3. 즉시 `loopId`, `runId` 반환
4. Worker는 Loop 진행 중 이벤트를 API Server로 콜백 (`POST /api/v2/events`)

**Configuration Parameters**:
- `maxRounds`: 최대 Round 수 (현재: 3 고정, 추후 변경 가능)
- `confThreshold`: Confidence threshold (PASS/FAIL 판정)
- `failThreshold`: Fail ratio 종료 조건
- `minFailCount`: 최소 FAIL 개수 (재학습 조건)
- `patience`: Early stopping patience
- `iouThreshold`: NMS IoU threshold (bbox 병합)

---

### 2.2 Get Loop Status

**Endpoint**: `GET /api/v2/loop/status/{loop_id}`

**Description**: Loop 진행 상태 조회 (Worker로 프록시)

**Request**:
```http
GET /api/v2/loop/status/loop_abc123
```

**Response**:
```json
{
  "loopId": "loop_abc123",
  "runId": "run_20250106_120000_xyz789",
  "status": "RUNNING",
  "stats": {
    "currentRound": 1,
    "totalRounds": 3,
    "roundHistory": [
      {
        "round": 0,
        "total": 1000,
        "passThree": 650,
        "passTwo": 200,
        "fail": 100,
        "miss": 50,
        "failMissRatio": 0.15
      }
    ],
    "latestFailMissRatio": 0.15,
    "results": {
      "note": "Results are built by /api/v2/results/* endpoints",
      "suggested": {
        "roundPreview": "/api/v2/results/round/{round_number}/preview?loopId=xxx",
        "finalPreview": "/api/v2/results/final/preview?loopId=xxx"
      }
    }
  }
}
```

**Status Values**:
- `STARTED`: Loop 시작됨
- `RUNNING`: 실행 중
- `COMPLETED`: 완료
- `FAILED`: 실패

**Round History**:
- `round`: Round 번호 (0, 1, 2)
- `total`: 전체 이미지 수
- `passThree`: 3개 모델 모두 PASS
- `passTwo`: 2개 PASS, 1개 FAIL
- `fail`: 1개 PASS, 2개 FAIL
- `miss`: 3개 모두 FAIL
- `failMissRatio`: (FAIL + MISS) / total

---

## 3. Event APIs

Worker가 Loop 진행 중 이벤트를 API Server로 콜백하고, Spring Boot 프론트엔드가 폴링으로 조회합니다.

### 3.1 Ingest Event (Worker Callback)

**Endpoint**: `POST /api/v2/events`

**Description**: Worker가 Loop 진행 중 이벤트 콜백

**Request**:
```json
{
  "eventType": "LOOP_STARTED",
  "runId": "run_20250106_120000_xyz789",
  "jobId": "job_20250106_120000",
  "message": "Loop started",
  "payload": {
    "loopId": "loop_abc123",
    "models": ["yolo", "model2", "model3"]
  }
}
```

**Response**:
```json
{
  "resultCode": "SUCCESS",
  "message": "Event received"
}
```

**Event Types**:
- `LOOP_STARTED`: Loop 시작
- `LOOP_DONE`: Loop 완료
- `LOOP_FAILED`: Loop 실패
- `ROUND_RESULT`: Round 결과
- `EXPORT_FINAL_READY`: Final export 준비
- `EXPORT_FINAL_DONE`: Final export 완료
- `EXPORT_FINAL_FAILED`: Final export 실패

**Storage**: `data/logs/events/{runId}/{timestamp}_{eventType}.json`

---

### 3.2 Get Latest Event

**Endpoint**: `GET /api/v2/events/latest`

**Description**: 최신 이벤트 조회 (Spring Boot 폴링용)

**Request**:
```http
GET /api/v2/events/latest?runId=run_20250106_120000_xyz789&eventType=LOOP_PROGRESS
```

**Query Parameters**:
- `runId`: string (required) - Run 식별자
- `eventType`: string (optional) - 이벤트 타입 필터

**Response**:
```json
{
  "resultCode": "SUCCESS",
  "data": {
    "receivedAt": "2025-01-06T12:00:00Z",
    "event": {
      "eventType": "LOOP_PROGRESS",
      "runId": "run_xxx",
      "data": {...}
    },
    "_fileName": "20250106_120000_123456_LOOP_PROGRESS.json"
  }
}
```

**Polling Pattern** (Spring Boot):
```javascript
// 2초마다 폴링
setInterval(() => {
  fetch(`/api/v2/events/latest?runId=${runId}`)
    .then(response => response.json())
    .then(data => {
      updateProgressBar(data.event.passCount, data.event.failCount);
      updateStatusMessage(data.event.message);
      if (data.event.exportRelPath) {
        showDownloadButton(data.event.exportRelPath);
      }
    });
}, 2000);
```

---

### 3.3 List Event Runs

**Endpoint**: `GET /api/v2/events/runs`

**Description**: 이벤트 Run 목록 조회 (최근순)

**Request**:
```http
GET /api/v2/events/runs?limit=50
```

**Response**:
```json
{
  "resultCode": "SUCCESS",
  "data": [
    {
      "runId": "run_20250106_120000_xyz789",
      "runPath": "/path/to/events/run_xxx",
      "updatedAt": "2025-01-06T12:30:00Z"
    }
  ]
}
```

---

### 3.4 Get Events (Pagination)

**Endpoint**: `GET /api/v2/events/get`

**Description**: 특정 Run의 이벤트 목록 조회 (Pagination)

**Request**:
```http
GET /api/v2/events/get?runId=run_20250106_120000_xyz789&offset=0&limit=200
```

**Response**:
```json
{
  "resultCode": "SUCCESS",
  "data": {
    "runId": "run_20250106_120000_xyz789",
    "total": 10,
    "offset": 0,
    "limit": 200,
    "items": [
      {
        "receivedAt": "2025-01-06T12:00:00Z",
        "event": {...},
        "_fileName": "..."
      }
    ]
  }
}
```

---

## 4. Export APIs

Round별 및 최종 결과를 ZIP 파일로 생성하고 다운로드합니다.

### 4.1 Export Round

**Endpoint**: `POST /api/v2/export/round`

**Description**: Round별 결과 Export (ZIP 생성)

**Request**:
```http
POST /api/v2/export/round?loopId=loop_abc123&runNumber=0
```

**Query Parameters**:
- `loopId`: string (required)
- `runNumber`: integer (required) - Round 번호 (0, 1, 2)

**Response**:
```json
{
  "resultCode": "SUCCESS",
  "message": "Round 0 exported successfully",
  "data": {
    "loopId": "loop_abc123",
    "runNumber": 0,
    "zipPath": "exports/loop_abc123/run_0.zip",
    "fileSize": 1048576
  }
}
```

**ZIP Structure**:
```
run_0.zip
├── PASS_THREE/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── labels/
│       ├── img001.txt
│       └── img002.txt
├── PASS_TWO/
│   ├── images/
│   └── labels/
├── FAIL/
│   ├── images/
│   └── labels/
└── MISS/
    └── images/
```

**Categories**:
- `PASS_THREE`: 3개 모델 모두 PASS (images + labels)
- `PASS_TWO`: 2개 PASS, 1개 FAIL (images + labels)
- `FAIL`: 1개 PASS, 2개 FAIL (images + labels)
- `MISS`: 3개 모두 FAIL (images only, no labels)

---

### 4.2 Download Round Export

**Endpoint**: `GET /api/v2/export/round/download`

**Description**: Round Export ZIP 다운로드

**Request**:
```http
GET /api/v2/export/round/download?loopId=loop_abc123&runNumber=0
```

**Response**:
- Content-Type: `application/zip`
- Content-Disposition: `attachment; filename="loop_loop_abc123_run_0.zip"`

---

### 4.3 Export Final

**Endpoint**: `POST /api/v2/export/final`

**Description**: 최종 결과 Export (전체 Round 통합)

**Request**:
```http
POST /api/v2/export/final?loopId=loop_abc123
```

**Response**:
```json
{
  "resultCode": "SUCCESS",
  "message": "Final results exported successfully",
  "data": {
    "loopId": "loop_abc123",
    "zipPath": "exports/loop_abc123/final.zip",
    "fileSize": 5242880,
    "summary": {
      "totalPass": 960,
      "totalFail": 30,
      "totalMiss": 10
    }
  }
}
```

**ZIP Structure**:
```
final.zip
├── PASS/              (모든 Round의 PASS_THREE + PASS_TWO 통합)
│   ├── images/
│   └── labels/
├── FAIL/              (최종 Round의 FAIL)
│   ├── images/
│   └── labels/
└── MISS/              (최종 Round의 MISS)
    └── images/
```

**Processing**:
1. 모든 Round의 `PASS_THREE/`, `PASS_TWO/`를 `PASS/`로 병합
2. 최종 Round의 `FAIL/`을 `FAIL/`로 복사
3. 최종 Round의 `MISS/`를 `MISS/`로 복사

---

### 4.4 Download Final Export

**Endpoint**: `GET /api/v2/export/final/download`

**Description**: 최종 Export ZIP 다운로드

**Request**:
```http
GET /api/v2/export/final/download?loopId=loop_abc123
```

**Response**:
- Content-Type: `application/zip`
- Content-Disposition: `attachment; filename="loop_loop_abc123_final.zip"`

---

## 5. Data Structures

### 5.1 Directory Structure

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
│       │   │   ├── images/
│       │   │   └── labels/
│       │   ├── FAIL/
│       │   │   ├── images/
│       │   │   └── labels/
│       │   └── MISS/
│       │       └── images/
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
│   ├── gt_yolo_xxx/
│   └── unlabeled_model2_xxx/
│
└── logs/
    └── events/                 # 이벤트 로그 JSON
        └── {runId}/
            ├── 20250106_120000_123456_LOOP_STARTED.json
            ├── 20250106_120001_234567_ROUND_RESULT.json
            └── 20250106_120030_345678_LOOP_DONE.json
```

### 5.2 Label Format (YOLO)

```
# YOLO format: class cx cy w h
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.15 0.25

# Normalized coordinates (0.0 ~ 1.0)
# cx, cy: center coordinates
# w, h: width, height
```

### 5.3 Ensemble Classification

각 이미지에 대해 3개 모델의 결과를 조합:

| YOLO | Model2 | Model3 | Category |
|------|--------|--------|----------|
| PASS | PASS   | PASS   | **PASS_THREE** |
| PASS | PASS   | FAIL   | **PASS_TWO** |
| PASS | FAIL   | PASS   | **PASS_TWO** |
| FAIL | PASS   | PASS   | **PASS_TWO** |
| PASS | FAIL   | FAIL   | **FAIL** |
| FAIL | PASS   | FAIL   | **FAIL** |
| FAIL | FAIL   | PASS   | **FAIL** |
| FAIL | FAIL   | FAIL   | **MISS** |

**PASS/FAIL Criteria**:
- PASS: Confidence ≥ threshold (default: 0.5)
- FAIL: Confidence < threshold

### 5.4 Round Repetition Strategy

```
Round 0:
  Input: 모든 Unlabeled 이미지 (1000장)
  Process: 3모델 병렬 추론 → 앙상블 분류
  Output: PASS_THREE, PASS_TWO, FAIL, MISS

Round 1:
  Input: 전체 Unlabeled 이미지 (1000장)
  Retrain: Round 0의 FAIL 데이터로 각 모델 재학습
  Process: 재학습된 모델로 전체 Unlabeled 재추론
  Output: PASS_THREE, PASS_TWO, FAIL, MISS

Round 2:
  Input: 전체 Unlabeled 이미지 (1000장)
  Retrain: Round 1의 FAIL 데이터로 각 모델 재학습
  Process: 재학습된 모델로 전체 Unlabeled 재추론
  Output: PASS_THREE, PASS_TWO, FAIL, MISS
```

**Current**: 3 Rounds 고정 (0, 1, 2)
**Future**: Fail ratio 기반 동적 종료 조건 추가 예정

---

## API Summary

### Total Endpoints: 28

| Category | Endpoints | Description |
|----------|-----------|-------------|
| **Data Ingest** | 18 | GT/Unlabeled 업로드 및 관리 (모델당 6개 × 3) |
| **Loop** | 2 | Loop 시작 및 상태 조회 |
| **Event** | 4 | 이벤트 수신 및 조회 |
| **Export** | 4 | Round별 및 최종 결과 Export |

### By Model

| Model | GT Upload | GT Register | GT Versions | Unlabeled Upload | Unlabeled Info |
|-------|-----------|-------------|-------------|------------------|----------------|
| YOLO | ✅ | ✅ | ✅ | ✅ | ✅ |
| Model2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Model3 | ✅ | ✅ | ✅ | ✅ | ✅ |

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
| 400 | Bad Request (Invalid parameters, path traversal, etc.) |
| 404 | Not Found (Loop, Run, Event, Export not found) |
| 500 | Internal Server Error (Processing failed) |
| 503 | Service Unavailable (Worker unreachable) |

### Worker Proxy Errors

Loop API에서 Worker 통신 실패 시:

```json
{
  "detail": {
    "message": "Worker unreachable: Connection refused",
    "url": "http://v1-worker:8011/api/v2/loop/start"
  },
  "status_code": 503
}
```

---

## Authentication & Security

### Current Version
- No authentication required (internal network)
- Path traversal protection (Event API)
- Input validation (Pydantic models)

### Future Enhancements
- API Key authentication
- Rate limiting
- Request logging
- CORS configuration

---

## Notes

1. **Worker Dependency**: Loop API는 Worker Server가 실행 중이어야 합니다
2. **File-Based Storage**: 이벤트는 JSON 파일로 저장되며 영구 보관됩니다
3. **Export Caching**: Export ZIP은 재생성 방지를 위해 캐시됩니다
4. **Polling Optimization**: `/events/latest`는 Spring Boot 폴링에 최적화되어 있습니다
5. **Model Independence**: 각 모델의 GT/Unlabeled 데이터는 완전히 독립적으로 관리됩니다

---

**Last Updated**: 2025-01-06
**Maintained By**: V2 Ensemble Team
