# 🛠️ AI 에이전트 개발 행동 강령 (YOLO 해커톤 통합 에디션)

> **[필독] 에이전트 구동 지침**
>
> 1. 본 프로젝트는 **풍력 터빈 블레이드 Dirt/Damage 객체 탐지(YOLO11)** 와 이를 서비스하는 **B2B 웹 프로덕트**를 통합 구축합니다.
> 2. **AI/ML 작업:** Apple M1 Pro(`device='mps'`)를 활용하며, 코드 수정 후 반드시 **1 Epoch 테스트**(`python train.py --test`)로 검증하십시오.
> 3. **설정 하드코딩 금지:** 모델·하이퍼파라미터·증강은 `configs/*.yaml`, 데이터 경로는 `data/data.yaml`에서 관리합니다.
> 4. **UI/UX 작업:** 프론트엔드 착수 시 루트의 `DESIGN.md`를 기준으로 LogPick Navy/Teal, Pretendard를 적용하십시오. *(현재 `DESIGN.md` 미작성 — Phase 3 착수 전 생성 필요)*

---

## 0. 프로젝트 현황 (2026-07 기준)

### Phase 진행 상태

| Phase | 내용 | 상태 |
| :--- | :--- | :---: |
| **Phase 1** | 데이터 분할 · EDA · 학습 · 검증 · 리포트 | 🟢 **대부분 완료** |
| **Phase 1 Test** | `predict.py` 단일 이미지 추론 | ⬜ **다음 작업** |
| **Phase 2** | FastAPI 백엔드 (`backend/`) | ⬜ 미착수 |
| **Phase 3** | Next.js 프론트 (`frontend-user/`) | ⬜ 미착수 |

### 학습 결과 (Validation 기준)

| 구분 | 모델 | mAP50 | mAP50-95 | 가중치 |
| :--- | :--- | ---: | ---: | :--- |
| **Baseline** | YOLO11n (Nano) | 0.538 | 0.317 | `runs/detect/baseline/weights/best.pt` |
| **최종 모델** | YOLO11s (Small) | 0.575 | 0.319 | `runs/detect/train/weights/best.pt` |
| **재검증** | YOLO11s | 0.574 | 0.318 | `runs/detect/val_final/` |

- **데이터:** Train/Val **8:2** 분할 (독립 Test 세트 없음 → Val 기준 평가)
- **클래스:** `dirt`(0), `damage`(1)
- **EDA:** Damage BBox 93%가 극소형(w,h < 0.2) — `runs/eda/` 참조

---

## 1. 프로젝트 구조

```
wind-turbine-yolo/
├── split_data.py          # Train/Val 8:2 분할 (배경 이미지·고아 라벨 처리)
├── eda.py                 # YOLO 라벨 EDA · 시각화
├── train.py               # YOLO11 본학습
├── val.py                 # best.pt Val 재검증
├── update_report.py       # report.md 자동 갱신
├── update_notion.py       # Notion 페이지 동기화 + 이미지 업로드
├── report.md              # 해커톤 루브릭 리포트 (자동 마커 포함)
├── TRAINING_CHECKLIST.md  # 학습 준비·추가 개발 체크리스트
├── configs/
│   ├── train.yaml         # 본학습 (YOLO11s, 50 epoch, batch 8)
│   ├── train_baseline.yaml # Baseline (YOLO11n, 20 epoch)
│   └── val.yaml           # Val 재검증 (val_final)
├── data/
│   ├── data.yaml          # 데이터셋 경로·클래스
│   ├── images/{train,val}
│   └── labels/{train,val}
├── runs/
│   ├── detect/            # YOLO 학습·검증 산출물
│   └── eda/               # EDA 그래프·eda_summary.yaml
├── .env                   # NOTION_TOKEN 등 (Git 제외) ← 실제 비밀값
└── .env.example           # 환경 변수 예시 (토큰 넣지 말 것)
```

---

## 2. 파이프라인 및 명령어 (Commands)

### ① AI 머신러닝 파이프라인 (YOLO11) — 권장 실행 순서

```bash
# 0. 의존성 (최초 1회)
pip3 install -r requirements.txt

# 1. 데이터 분할 (최초 1회 — 파일 이동 주의)
python3 split_data.py

# 2. EDA (학습 전·후 모두 가능)
python3 eda.py                    # runs/eda/ 생성 + report.md EDA 섹션 갱신

# 3. 본학습
python3 train.py                  # configs/train.yaml
python3 train.py --test           # 1 Epoch 파이프라인 스모크 테스트

# 4. Baseline (비교군)
python3 train.py --config configs/train_baseline.yaml --no-report

# 5. Val 재검증
python3 val.py                    # runs/detect/val_final/

# 6. 리포트·Notion 반영
python3 update_report.py
python3 update_notion.py          # .env 필수 (아래 §5 참조)
```

**한 번에 Notion까지:** `python3 update_notion.py`  
→ EDA 없으면 자동 생성 → `update_report.py` → Notion 업로드(로컬 이미지 File Upload API)

### ② Phase 1 Test — 추론 (미구현 · 다음 작업)

- **목표:** `predict.py` — `best.pt`로 단일 이미지 BBox 추론 · 결과 이미지/JSON 저장
- **OSS:** Ultralytics YOLO + OpenCV (`references.md` S2)

### ③ 백엔드 (Backend - FastAPI) — 미착수

- **역할:** `best.pt` 로드 → 이미지 업로드 → BBox·클래스 JSON 반환
- **실행:** `uvicorn backend.main:app --reload`
- **검증:** Swagger UI `http://localhost:8000/docs`

### ④ 프론트엔드 (Frontend - Next.js) — 미착수

- **역할:** LogPick 테마 B2B 데모 UI (좌: 업로드 / 우: 탐지 결과·카운트)
- **실행:** `npm install` → `npm run dev`

---

## 3. 핵심 설정 규칙

### 도메인 맞춤형 AI (`configs/train.yaml`)

| 항목 | 값 | 이유 |
| :--- | :--- | :--- |
| `device` | `mps` | Apple M1 GPU |
| `flipud` | `0.0` | 풍력 터빈 블레이드 상하 반전 **절대 금지** |
| `batch` | `8` | M1 16GB RAM — 32는 스왑 유발 |
| `workers` | `0` | Mac+MPS 메모리 부담 감소 |
| `patience` | `10` | Early stopping |
| `epochs` | `50` (본학습) / `20` (Baseline) | |

- **OOM 대응:** `train.py` · `val.py` — batch 8→4→2 자동 재시도
- **project 경로:** 스크립트가 프로젝트 루트 기준 **절대 경로**로 고정 (`runs/detect/`)

### 평가 지표

- **사용:** mAP50, mAP50-95, Precision, Recall, F1 곡선, Confusion Matrix
- **한계:** Hold-out Test 없음 — 발표 시 **"Validation 기준"** 명시
- **알려진 이슈:** Damage→Background FN 다수 (혼동행렬 참조)

---

## 4. 리포트·Notion 자동화

### `report.md` 자동 갱신 마커

| 마커 | 내용 |
| :--- | :--- |
| `report:auto:eda` | EDA 클래스 분포·인사이트·그래프 |
| `report:auto:run-summary` | 최종 학습·재검증 요약 |
| `report:auto:exp-comparison` | Baseline / EXP 비교 표 |
| `report:auto:metrics-visuals` | Loss/mAP · Confusion Matrix · F1 |
| `report:auto:predictions` | Val 예측 BBox 이미지 |

- **`update_report.py`:** `runs/detect/` + `runs/eda/` 스캔 → `report.md` 갱신
- **`update_notion.py`:** `report.md` → Notion 블록 변환 + **로컬 이미지 업로드**
- **`eda.py`:** 완료 시 `update_report.py` 자동 호출

### Notion 환경 설정 (필수)

```bash
cp .env.example .env
# .env 파일에만 토큰 입력 ( .env.example 에 넣지 말 것 )
```

```
NOTION_TOKEN=ntn_...
NOTION_PAGE_ID=38fb8ed24414801e9db4c45637297082
```

- Notion 페이지 → `···` → **연결** → Integration 추가 필수
- 스크립트는 **`.env`만** 읽음 (`.env.example`은 무시)

---

## 5. 중간 산출물 보존 (Visualization & Logging)

학습·검증·EDA 과정의 시각적 결과물은 **발표 핵심 근거**입니다. 삭제하지 마십시오.

| 경로 | 내용 |
| :--- | :--- |
| `runs/detect/train/` | `results.csv`, `results.png`, `weights/best.pt` |
| `runs/detect/baseline/` | Baseline 학습 결과 |
| `runs/detect/val_final/` | 재검증 메트릭·혼동행렬·예측 이미지 |
| `runs/eda/` | 클래스 분포·BBox 크기 EDA 그래프 |

**에이전트 역할:** 실험 완료 시 `update_report.py` → (선택) `update_notion.py` 실행.

---

## 6. 프론트엔드 UI/UX (Phase 3 — Based on DESIGN.md)

`DESIGN.md` 작성 후 아래 규칙을 적용합니다.

1. **테마:** LogPick B2B SaaS — 차분·고밀도, 과도한 장식 배제
2. **컬러:** Primary Teal `#0D9488`, Navy `#1E3A5F` / 폰트: `Pretendard`
3. **레이아웃:** 좌측 원본 업로드 · 우측 YOLO 탐지 결과·객체 수

---

## 7. 코딩 스타일 · 에러 핸들링

- **YAML 분리:** 학습 설정을 Python에 하드코딩하지 않음
- **AI OOM:** batch 축소 → `data.yaml` 경로 점검
- **Web (Phase 2~3):** CORS·API 타임아웃 처리, 핀포인트 디버깅
- **비밀값:** API 토큰·포트는 `.env` only (Git 커밋 금지)

---

## 8. Git 규칙

**`.gitignore` 대상:** `data/`, `*.pt`, `.env`, `node_modules/`, `runs/` (선택)

**커밋 예시:**
- `feat(ml): eda.py EDA 시각화 추가`
- `feat(backend): YOLO11 추론 엔드포인트 구현`
- `docs: report.md Baseline 비교 반영`

---

## 9. 참조 문서

| 문서 | 용도 |
| :--- | :--- |
| `references.md` | OSS 스택 · Phase 매핑 |
| `report.md` | 해커톤 루브릭 리포트 |
| `TRAINING_CHECKLIST.md` | 학습 준비·추가 개발 체크리스트 |
| `.cursorrules` | YOLO + M1 mps 규칙 |
