# V2 Ensemble API Endpoints

**Base URL**: `/api/v2`

---

## Quick Reference

### 📦 Data Ingest APIs (18)

#### YOLO Model (6)
```
POST   /api/v2/yolo/gt/upload             # GT ZIP 업로드
POST   /api/v2/yolo/gt/register           # GT 등록
GET    /api/v2/yolo/gt/versions           # GT 버전 목록
POST   /api/v2/yolo/unlabeled/upload      # Unlabeled 업로드
GET    /api/v2/yolo/unlabeled/info        # Unlabeled 정보
```

#### Model2 (6)
```
POST   /api/v2/model2/gt/upload
POST   /api/v2/model2/gt/register
GET    /api/v2/model2/gt/versions
POST   /api/v2/model2/unlabeled/upload
GET    /api/v2/model2/unlabeled/info
```

#### Model3 (6)
```
POST   /api/v2/model3/gt/upload
POST   /api/v2/model3/gt/register
GET    /api/v2/model3/gt/versions
POST   /api/v2/model3/unlabeled/upload
GET    /api/v2/model3/unlabeled/info
```

---

### 🔄 Loop APIs (2)

```
POST   /api/v2/loop/start                 # Loop 시작 (Worker 프록시)
GET    /api/v2/loop/status/{loop_id}      # Loop 상태 조회
```

---

### 📡 Event APIs (4)

```
POST   /api/v2/events                     # 이벤트 수신 (Worker 콜백)
GET    /api/v2/events/runs                # Run 목록 조회
GET    /api/v2/events/get                 # 이벤트 목록 (pagination)
GET    /api/v2/events/latest              # 최신 이벤트 (폴링용)
```

---

### 📤 Export APIs (4)

```
POST   /api/v2/export/round               # Round Export 생성
GET    /api/v2/export/round/download      # Round ZIP 다운로드
POST   /api/v2/export/final               # Final Export 생성
GET    /api/v2/export/final/download      # Final ZIP 다운로드
```

---

## Detailed Endpoints

### Data Ingest

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| POST | `/yolo/gt/upload` | GT ZIP 업로드 | `file`, `sourceName`, `datasetName` |
| POST | `/yolo/gt/register` | GT 등록 | `ingestId`, `copyMode`, `strict` |
| GET | `/yolo/gt/versions` | GT 버전 목록 | - |
| POST | `/yolo/unlabeled/upload` | Unlabeled 업로드 | `file`, `datasetName` |
| GET | `/yolo/unlabeled/info` | Unlabeled 정보 | - |

*Model2, Model3도 동일한 구조*

---

### Loop

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| POST | `/loop/start` | Loop 시작 | `EnsembleLoopRequest` | `EnsembleLoopResponse` |
| GET | `/loop/status/{loop_id}` | Loop 상태 조회 | - | `LoopStatusResponse` |

**Request Example**:
```json
{
  "models": ["yolo", "model2", "model3"],
  "configOverride": {
    "maxRounds": 3,
    "confThreshold": 0.5
  }
}
```

---

### Event

| Method | Endpoint | Description | Query Params |
|--------|----------|-------------|--------------|
| POST | `/events` | 이벤트 수신 | - |
| GET | `/events/runs` | Run 목록 | `limit` |
| GET | `/events/get` | 이벤트 목록 | `runId`, `offset`, `limit` |
| GET | `/events/latest` | 최신 이벤트 | `runId`, `eventType` |

**Event Types**:
- `LOOP_STARTED`
- `LOOP_DONE`
- `LOOP_FAILED`
- `ROUND_RESULT`
- `EXPORT_FINAL_READY`
- `EXPORT_FINAL_DONE`
- `EXPORT_FINAL_FAILED`

---

### Export

| Method | Endpoint | Description | Query Params |
|--------|----------|-------------|--------------|
| POST | `/export/round` | Round Export 생성 | `loopId`, `runNumber` |
| GET | `/export/round/download` | Round ZIP 다운로드 | `loopId`, `runNumber` |
| POST | `/export/final` | Final Export 생성 | `loopId` |
| GET | `/export/final/download` | Final ZIP 다운로드 | `loopId` |

---

## Usage Workflows

### 1. 데이터 준비 (GT + Unlabeled)

```bash
# 1) YOLO GT 업로드
POST /api/v2/yolo/gt/upload
  file: gt_yolo.zip

# 2) YOLO GT 등록
POST /api/v2/yolo/gt/register?ingestId=gt_yolo_xxx

# 3) YOLO Unlabeled 업로드
POST /api/v2/yolo/unlabeled/upload
  file: unlabeled_yolo.zip

# 4-6) Model2, Model3도 동일하게 반복
```

---

### 2. Loop 실행

```bash
# 1) Loop 시작
POST /api/v2/loop/start
{
  "models": ["yolo", "model2", "model3"],
  "configOverride": {
    "maxRounds": 3,
    "confThreshold": 0.5
  }
}
→ Response: { "loopId": "loop_abc123", "runId": "run_xyz789" }

# 2) 상태 폴링 (Spring Boot)
GET /api/v2/events/latest?runId=run_xyz789
→ 2초마다 반복

# 3) Loop 상태 조회
GET /api/v2/loop/status/loop_abc123
```

---

### 3. 결과 Export

```bash
# 1) Round 0 Export
POST /api/v2/export/round?loopId=loop_abc123&runNumber=0

# 2) Round 0 다운로드
GET /api/v2/export/round/download?loopId=loop_abc123&runNumber=0

# 3) Final Export
POST /api/v2/export/final?loopId=loop_abc123

# 4) Final 다운로드
GET /api/v2/export/final/download?loopId=loop_abc123
```

---

## cURL Examples

### GT Upload
```bash
curl -X POST http://localhost:8010/api/v2/yolo/gt/upload \
  -F "file=@gt_yolo.zip" \
  -F "sourceName=client_A" \
  -F "datasetName=202501_batch1"
```

### GT Register
```bash
curl -X POST "http://localhost:8010/api/v2/yolo/gt/register?ingestId=gt_yolo_xxx&copyMode=symlink&strict=false"
```

### Loop Start
```bash
curl -X POST http://localhost:8010/api/v2/loop/start \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["yolo", "model2", "model3"],
    "configOverride": {
      "maxRounds": 3,
      "confThreshold": 0.5
    }
  }'
```

### Event Polling
```bash
curl "http://localhost:8010/api/v2/events/latest?runId=run_xyz789"
```

### Export Round
```bash
curl -X POST "http://localhost:8010/api/v2/export/round?loopId=loop_abc123&runNumber=0"
```

### Download Export
```bash
curl -O "http://localhost:8010/api/v2/export/round/download?loopId=loop_abc123&runNumber=0"
```

---

## Python SDK Example

```python
import requests

# 1. GT 업로드
with open("gt_yolo.zip", "rb") as f:
    response = requests.post(
        "http://localhost:8010/api/v2/yolo/gt/upload",
        files={"file": f},
        data={"sourceName": "client_A"}
    )
    ingest_id = response.json()["ingestId"]

# 2. GT 등록
response = requests.post(
    "http://localhost:8010/api/v2/yolo/gt/register",
    params={"ingestId": ingest_id, "copyMode": "symlink"}
)
print(response.json())

# 3. Loop 시작
response = requests.post(
    "http://localhost:8010/api/v2/loop/start",
    json={
        "models": ["yolo", "model2", "model3"],
        "configOverride": {
            "maxRounds": 3,
            "confThreshold": 0.5
        }
    }
)
loop_data = response.json()
loop_id = loop_data["loopId"]
run_id = loop_data["runId"]

# 4. 이벤트 폴링
import time
while True:
    response = requests.get(
        f"http://localhost:8010/api/v2/events/latest",
        params={"runId": run_id}
    )
    event = response.json()

    if event["data"]:
        event_type = event["data"]["event"]["eventType"]
        print(f"Event: {event_type}")

        if event_type == "LOOP_DONE":
            break

    time.sleep(2)

# 5. Final Export
response = requests.post(
    "http://localhost:8010/api/v2/export/final",
    params={"loopId": loop_id}
)
print(response.json())

# 6. 다운로드
response = requests.get(
    "http://localhost:8010/api/v2/export/final/download",
    params={"loopId": loop_id}
)
with open("final_export.zip", "wb") as f:
    f.write(response.content)
```

---

## Environment Variables

```bash
# Worker Server URL
WORKER_BASE_URL=http://v1-worker:8011

# Event Storage
V2_EVENTS_ROOT=/mnt/nas/v2_events

# Export Storage
V2_EXPORTS_ROOT=/mnt/nas/v2_exports
```

---

**Total Endpoints**: 28
- Data Ingest: 18
- Loop: 2
- Event: 4
- Export: 4
