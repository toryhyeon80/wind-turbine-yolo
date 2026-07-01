# 학습 준비 체크리스트 (wind-turbine-yolo)

> 풍력 터빈 블레이드 **Dirt / Damage** 객체 탐지(YOLO11) 프로젝트 기준  
> 발표·리포트용 근거 정리 — 최종 갱신: 2026-07-01

**문서 구성**

1. **기본 ML 체크리스트** — 데이터 / 모델 학습 / 평가 (18항목)
2. **체크리스트 외 추가 개발** — 체크리스트 없이 진행한 파이프라인·도메인·자동화 작업

### 상태 범례

| 표기 | 의미 |
| :--- | :--- |
| `[x]` | 완료 |
| `[x]` 🟡 | 부분 완료 (한계·보완 사항 있음) |
| `[ ]` | 미완 / 진행 중 |

### 근거 파일 빠른 참조

| 주제 | 경로 |
| :--- | :--- |
| 데이터 분할 | `split_data.py`, `data/data.yaml` |
| 학습 설정 | `configs/train.yaml`, `configs/train_baseline.yaml` |
| 학습 로그 | `runs/detect/train/results.csv`, `results.png` |
| 최종 검증 | `runs/detect/val_final/val_metrics.yaml` |
| 시각화 | `runs/detect/val_final/confusion_matrix.png`, `val_batch*_pred.jpg` |
| 종합 리포트 | `report.md` |

---

## 데이터

- [x] **데이터 수집 방법을 결정했는가?** (공개 데이터셋 / 크롤링 / API / 직접 수집 등)
  - **방법:** 드론 촬영 풍력 터빈 블레이드 이미지 + YOLO 형식 라벨링 (직접 수집·가공)
  - **클래스:** `dirt`(0), `damage`(1) — `data/data.yaml`
  - **발표 멘트:** 직접 수집한 블레이드 이미지를 YOLO 탐지 형식으로 구축

- [x] **충분한 양의 데이터를 확보했는가?**
  - **규모:** Train 10,776장 / Val 2,694장 (총 13,470장)
  - **구성:** 라벨 있는 객체 샘플 + 배경(라벨 없음) 이미지 포함
  - **발표 멘트:** 1만 장 이상 규모, 배경 이미지로 오탐 억제 학습

- [x] **데이터의 기본 구조(컬럼 수, 데이터 타입 등)를 파악했는가?**
  - **이미지:** `.jpg` 등 래스터 이미지
  - **라벨:** `.txt` — `class_id x_center y_center width height` (YOLO 정규화 좌표)
  - **매칭:** 파일명 stem 기준 이미지–라벨 1:1 (`split_data.py`)
  - **발표 멘트:** 이미지–라벨 쌍 구조와 2클래스 BBox 형식 파악 완료

- [x] **데이터 증강이 필요한 경우 증강 방법을 결정했는가?**
  - **설정:** `configs/train.yaml` — `flipud=0.0`(상하 반전 금지), `fliplr`, HSV, mosaic, mixup, erasing
  - **Baseline:** `configs/train_baseline.yaml` — 동일 도메인 규칙, 증강 최소화
  - **발표 멘트:** 풍력 터빈 도메인에 맞춘 증강 설계 (안개·빛 반사·각도 다양성)

- [x] 🟡 **학습 / 검증 / 테스트 데이터셋으로 분리했는가?**
  - **완료:** Train : Val = **8 : 2** (`split_data.py`, seed=42)
  - **미구축:** 독립 **Test** 세트 없음 (Val 기준 평가)
  - **경로:** `data/images/train`, `data/images/val`
  - **발표 멘트:** Train·Val 8:2 분할 완료. 최종 성능은 Validation 기준

- [x] **데이터 정규화를 수행했는가?**
  - **방식:** 탐지 과제 — 픽셀 스케일링·리사이즈(`imgsz: 640`)는 Ultralytics가 학습 시 자동 처리
  - **설정:** `configs/train.yaml` → `imgsz: 640`
  - **발표 멘트:** 표 형식 정규화 대신 YOLO 파이프라인에서 이미지 정규화 자동 수행

---

## 모델 학습

- [x] **문제 유형에 적합한 모델을 선택했는가?**
  - **문제:** 객체 탐지 (Bounding Box)
  - **최종 모델:** YOLO11s (Small) — `configs/train.yaml`
  - **발표 멘트:** BBox 기반 Dirt/Damage 탐지에 YOLO11 선택

- [x] **학습 중 과적합(Overfitting) 여부를 모니터링했는가?**
  - **모니터링:** 매 Epoch Train/Val Loss·mAP 기록
  - **산출물:** `runs/detect/train/results.csv`, `results.png`
  - **발표 멘트:** Train·Val Loss 동시 추적으로 과적합 징후 모니터링

- [x] 🟡 **과적합 방지 기법을 적용했는가?**
  - **적용:** Early stopping (`patience: 10`), Data Augmentation, `close_mosaic: 10`, Cosine LR
  - **프레임워크 기본:** L2 (`weight_decay: 0.0005`)
  - **미적용:** Dropout, L1 커스텀 설계
  - **발표 멘트:** 조기 종료·증강·학습률 스케줄로 일반화 확보

- [x] 🟡 **하이퍼파라미터 튜닝을 수행했는가?**
  - **튜닝 항목:** 모델 크기(Nano→Small), Epoch(50), Batch(8), 증강 강도, patience
  - **문서화:** `report.md` EXP 1~3
  - **한계:** EXP별 독립 run 폴더·수치 비교는 부분적 (본학습 `train` 중심)
  - **발표 멘트:** 모델 스케일·증강·학습 스케줄 실험 수행

- [x] **문제에 적절한 평가 지표를 선정하였는가?**
  - **지표:** mAP50, mAP50-95, Precision, Recall, F1 곡선, Confusion Matrix
  - **산출물:** `runs/detect/val_final/`
  - **최종 수치 (Val):** mAP50 ≈ 0.574, mAP50-95 ≈ 0.318
  - **발표 멘트:** 객체 탐지 표준 지표(mAP·P/R·F1·혼동행렬) 사용

- [x] 🟡 **모델이 잘못 출력한 데이터 포인트를 확인하였는가?**
  - **시각 확인:** `val_final/val_batch0_pred.jpg` 등 예측 오버레이
  - **정량:** `confusion_matrix.png` — Dirt↔Damage 혼동 낮음, Damage→Background FN 다수
  - **보완 여지:** 클래스별 FN/FP 수치 표 정리
  - **발표 멘트:** 예측 시각화·혼동행렬로 오류 패턴 확인

---

## 평가

- [ ] **문제에 적합한 탐지 모델과 비교용 Baseline 모델을 선정했는가?**
  - **선정:** 최종 YOLO11s vs Baseline YOLO11n — `configs/train_baseline.yaml`
  - **진행:** Baseline 학습 진행 중 (`runs/detect/baseline/`, 20 Epoch)
  - **완료 후:** `python3 update_report.py` → `report.md` 비교 표 자동 갱신
  - **발표 멘트:** Small vs Nano 비교 설계. Baseline 완료 후 정량 비교 예정

- [x] 🟡 **테스트 데이터로 최종 성능을 확인했는가?** (학습 완료 후)
  - **완료:** 본학습 완료 (`runs/detect/train/weights/best.pt`)
  - **검증:** `python3 val.py` → `val_final` (Val 세트 재평가)
  - **미구축:** 독립 Test 세트 없음 → **Validation 기준** 최종 성능
  - **발표 멘트:** Val 기준 mAP50 0.575 확인. Hold-out Test는 미구축

- [x] **학습 로그(Loss, mAP)를 통해 성능을 확인했는가?** (학습 중)
  - **로그:** `runs/detect/train/results.csv` (50 Epoch)
  - **그래프:** `runs/detect/train/results.png`
  - **자동화:** `update_report.py`, `update_notion.py`
  - **발표 멘트:** Epoch별 Loss·mAP 로그로 학습 과정 검증

- [x] **학습률 감소 전략을 적용했는가?**
  - **설정:** `cos_lr: true`, `warmup_epochs: 3`
  - **근거:** `configs/train.yaml`, `runs/detect/train/args.yaml`
  - **발표 멘트:** Cosine LR + Warmup 적용

- [x] **학습률, 배치 크기 등 하이퍼파라미터를 설정했는가?**
  - **설정 파일:** `configs/train.yaml` (YAML 분리, 하드코딩 없음)
  - **주요 값:** batch 8, epochs 50, imgsz 640, device mps, patience 10
  - **OOM 대응:** `train.py` — batch 8→4→2 자동 재시도
  - **발표 멘트:** YAML 기반 하이퍼파라미터 관리 + M1 MPS 가속

- [x] 🟡 **오탐(False Positive)·미탐(False Negative)을 분석했는가?**
  - **분석:** Confusion Matrix — Damage→Background FN 866건 (핵심 이슈)
  - **정성:** `report.md` — 소형 Damage FN, Dirt/Damage 클래스 혼동 낮음
  - **보완 여지:** 클래스별 Precision/Recall 수치 표 추가
  - **발표 멘트:** 혼동행렬로 FP/FN 패턴 분석. Damage 미탐이 주요 개선 포인트

---

## 체크리스트 외 추가 개발 항목

> 기본 ML 체크리스트 없이 개발을 시작했기 때문에, 아래 항목들은 **체크리스트에 없지만 실제로 구현·진행된 작업**입니다.

### A. 파이프라인·자동화 (엔지니어링)

| # | 항목 | 상태 | 근거 | 발표 포인트 |
| :-: | :--- | :---: | :--- | :--- |
| A1 | 데이터 분할 자동화 (배경 이미지·고아 라벨 처리) | ✅ | `split_data.py` | 재현 가능한 전처리 스크립트 |
| A2 | YAML 기반 설정 분리 (하드코딩 금지) | ✅ | `data/data.yaml`, `configs/*.yaml` | 실험 설정 변경·추적 용이 |
| A3 | OOM 시 batch 자동 축소 | ✅ | `train.py`, `val.py` | M1 16GB 환경 안정 학습 |
| A4 | 학습 후 리포트 자동 갱신 | ✅ | `update_report.py` | mAP·그래프·이미지 자동 반영 |
| A5 | Notion 연동 + 이미지 업로드 | ✅ | `update_notion.py`, `.env.example` | 협업·발표 자료 자동 동기화 |
| A6 | 별도 재검증 파이프라인 | ✅ | `val.py`, `configs/val.yaml` | `best.pt` 공식 Val 재평가 |
| A7 | 1 Epoch 파이프라인 스모크 테스트 | ✅ | `train.py --test` | 학습 전 환경·데이터 검증 |
| A8 | Git·의존성·시크릿 관리 | ✅ | `.gitignore`, `requirements.txt` | 데이터·가중치 제외, 토큰 분리 |

### B. 도메인·환경 특화 (풍력 터빈 / M1 Mac)

| # | 항목 | 상태 | 근거 | 발표 포인트 |
| :-: | :--- | :---: | :--- | :--- |
| B1 | 상하 반전 금지 (`flipud=0`) | ✅ | `configs/train.yaml` | 블레이드 중력·촬영 방향 도메인 지식 반영 |
| B2 | 배경(negative) 이미지 학습 포함 | ✅ | `split_data.py` | 오탐 억제용 negative sample |
| B3 | Apple M1 GPU 가속 (`device=mps`) | ✅ | 전 학습·검증 스크립트 | 온디바이스(드론) 환경과 동일 계열 |
| B4 | 메모리 스왑 대응 (batch 8, workers 0) | ✅ | `configs/train.yaml` | Mac 실환경 OOM·스왑 방지 |
| B5 | Baseline 전용 설정 분리 | ✅ | `configs/train_baseline.yaml` | Nano vs Small 공정 비교 설계 |

### C. 발표·해커톤 대응 (문서·시각화)

| # | 항목 | 상태 | 근거 | 발표 포인트 |
| :-: | :--- | :---: | :--- | :--- |
| C1 | 해커톤 루브릭 1~3 섹션 리포트 | ✅ | `report.md` | 데이터·비교·실험·평가 체계적 정리 (섹션 1~4) |
| C2 | EXP 1~3 실험 로그 문서화 | 🟡 | `report.md` 섹션 3 | 스케일업·증강·하이퍼파라미터 실험 서술 |
| C3 | Loss/mAP·예측 이미지 자동 삽입 | ✅ | `update_report.py` | 시각 근거 자동 축적 |
| C4 | Baseline vs 최종 비교 프레임 | ⏳ | `update_report.py` | Baseline 완료 후 % 향상 자동 계산 |
| C5 | 학습 준비 체크리스트 문서화 | ✅ | `TRAINING_CHECKLIST.md` | 발표용 준비 과정 근거 정리 |

### D. 평가 심화 (기본 체크리스트 보완)

| # | 항목 | 상태 | 근거 | 발표 포인트 |
| :-: | :--- | :---: | :--- | :--- |
| D1 | Val 메트릭 이중 저장 (YAML + CSV) | ✅ | `val_metrics.yaml`, `results.csv` | 리포트·Notion 도구 호환 |
| D2 | 클래스별 혼동 분석 (Dirt / Damage) | 🟡 | `confusion_matrix.png` | 클래스 간 혼동은 낮음 |
| D3 | Damage→Background FN 이슈 식별 | 🟡 | 혼동행렬, `report.md` | 미탐이 핵심 개선 포인트 |
| D4 | 클래스별 mAP (dirt / damage) | 🟡 | Val 터미널 로그 | 리포트 표 미정리 |

### E. Phase 2·3 (아직 미착수 — `references.md` 기준)

| # | 항목 | 상태 | 예정 경로 | 비고 |
| :-: | :--- | :---: | :--- | :--- |
| E1 | 추론 스크립트 (`predict.py`) | ⬜ | Phase 1 Test | 단일 이미지 BBox 추론 |
| E2 | FastAPI 백엔드 | ⬜ | `backend/` | `best.pt` → JSON API |
| E3 | Next.js 프론트엔드 | ⬜ | `frontend-user/` | LogPick 테마 데모 UI |

---

## 전체 개발 맵 (한눈에 보기)

```
[기본 ML 체크리스트]          [체크리스트 외 추가 개발]
─────────────────────        ──────────────────────────
데이터 수집·구조·증강    →    split_data 자동화, 배경 이미지, flipud=0
Train/Val 분할          →    YAML 설정 분리, seed 고정
모델 선택·학습          →    MPS 가속, OOM fallback, --test 모드
과적합·하이퍼파라미터    →    Baseline 설정 분리, EXP 문서화
평가 지표·FP/FN         →    val.py 재검증, 혼동행렬 심화 분석
Baseline 비교           →    update_report 자동 비교표
(없음)                  →    report.md, Notion 자동화, Git 관리
(없음)                  →    Phase 2 API, Phase 3 UI (예정)
```

---

## 발표용 한 줄 요약

| 구분 | 내용 |
| :--- | :--- |
| **기본 체크리스트** | 18항목 중 완료 10 · 부분 5 · 진행 1 · 미착수 2(Phase 2·3) |
| **추가 개발** | 파이프라인 자동화 8 · 도메인 특화 5 · 문서·시각화 5 · 평가 심화 4 |
| **강점 (발표 강조)** | ML 필수 항목 + **재현 가능한 파이프라인·도메인 규칙·자동 리포트** |
| **진행 중** | Baseline(YOLO11n) 학습 → 완료 후 `update_report.py` |
| **한계·향후** | Test 세트 없음, Damage FN 보완, Phase 2 API·Phase 3 UI |
