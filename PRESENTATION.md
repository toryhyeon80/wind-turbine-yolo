# 🌪️ 풍력 발전기 스마트 예방 정비 시스템 — 15분 발표 PPT 초안

> **용도:** Gamma AI 슬라이드 제작용 · **총 발표 시간 15분** (Live Demo 2분 포함)  
> **슬라이드:** 18장 + 백업 2장  
> **구성:** Phase 1 (ML) + Phase 2·3 (데모) + 심화 목표 + 마무리  
> **Phase 1 상세:** `report.md` / Notion · **본 PPT:** 발표·데모·확장 중심

---

## 시간 배분 요약

| 구간 | 슬라이드 | 시간 |
|------|----------|------|
| 도입 | 1~3 | 2분 |
| 데이터·EDA | 4~6 | 3분 |
| 모델·학습·실험 | 7~9 | 3분 |
| 성능·평가 | 10~12 | 3분 |
| **Live Demo** | 13~14 | **2.5분** |
| 심화·로드맵 | 15~16 | 2.5분 |
| 마무리 | 17~18 | 1.5분 |

---

## Slide 1 · 표지 (0:30)

**제목:** 🌪️ 풍력 발전기 스마트 예방 정비 시스템

**부제:** YOLO11 기반 블레이드 Dirt/Damage 탐지 & 웹 데모

**하단**
- 팀명 / 발표자
- GitHub: https://github.com/toryhyeon80/wind-turbine-yolo
- Notion 리포트 (링크)

**발표 멘트**
> 드론으로 촬영한 풍력 터빈 블레이드에서 **오염(Dirt)** 과 **손상(Damage)** 을 AI로 자동 탐지하고, **웹 데모까지** 구현한 결과를 공유합니다.

---

## Slide 2 · 문제 정의 & 배경 (0:45)

**제목:** 왜 필요한가? — 예방 정비의 과제

**Bullet**
- 풍력 블레이드 **표면 결함** → 효율 저하·안전 리스크
- 드론 점검 → **수만 장** 이미지 · **육안 검수** 한계
- 필요: **자동 BBox 탐지** → 1차 스크린닝 → 검수자 확인

**목표 (한 줄)**
> Dirt(0) + Damage(1) **2클래스 객체 탐지** (YOLO11)

**발표 멘트**
> 점검 이미지가 많고 Damage는 **매우 작게** 나타나 탐지가 어렵습니다. YOLO로 **1차 자동 탐지** 후 사람이 확인하는 워크플로를 목표로 했습니다.

---

## Slide 3 · 프로젝트 파이프라인 (1:00)

**제목:** 전체 아키텍처 — Phase 1 → 2 → 3

**다이어그램**

```
[데이터] split_data → EDA → train/val
              ↓
[Phase 1] YOLO11s 학습 → val.py → predict.py
              ↓
[Phase 2] FastAPI  POST /api/v1/predict
              ↓
[Phase 3] Streamlit 웹 데모 (Live Demo)
              ↓
[자동화] report.md · Notion 동기화
```

**환경**
- Apple M1 Pro · **device=mps**
- 클래스: **dirt(0), damage(1)**

**발표 멘트**
> 데이터 분할부터 학습·검증·추론, API·웹 데모, 리포트 자동화까지 **End-to-End 파이프라인**을 구축했습니다.

---

## Slide 4 · 데이터셋 (1:00)

**제목:** 데이터셋 구축 — 13,470장

**표**

| 항목 | 수량 | 비율 |
|------|------|------|
| **전체** | **13,470장** | 100% |
| Train | 10,776 | 80% |
| Val | 2,694 | 20% |

**구성**

| 구분 | Train | Val | 합계 |
|------|------:|----:|-----:|
| 라벨 있음 (BBox) | 2,401 | 594 | 2,995 |
| **배경** (negative) | 8,375 | 2,100 | 10,475 |

**Bullet**
- 드론 촬영 + YOLO 형식 라벨 · `split_data.py` (seed=42)
- **배경 이미지** 포함 → 오탐 억제 학습
- **Test 세트:** 미구축 → **Val 기준** 평가 (한계 명시)

**캡처 제안:** `runs/eda/class_distribution.png`

**발표 멘트**
> 1만 3천 장 규모이며, **배경 1만 장**을 넣어 ‘아무것도 없는 이미지’도 학습했습니다. Test는 일정상 Val로 평가했고, **향후 Test 분리**가 과제입니다.

---

## Slide 5 · EDA 인사이트 (1:00)

**제목:** EDA — 데이터가 모델 설계를 이끈다

**핵심 수치**
- 총 BBox **9,351개**
- **Dirt : Damage ≈ 1 : 15** (581 vs 8,770)
- Damage BBox **93.3%** 극소형 (w,h < 0.2)

**설계 반영**

| EDA | 모델/학습 결策 |
|-----|----------------|
| 극소형 Damage | YOLO11 **Small**, mosaic 유지 |
| 클래스 불균형 | 도메인 증강, 2클래스 유지 |
| 블레이드 방향 | **flipud=0** (상하 반전 금지) |

**캡처 제안:** `runs/eda/bbox_size_distribution.png`, `class_distribution.png`

**발표 멘트**
> Damage가 **극소형·다수**라 Nano보다 **Small**이 필요하고, 풍력 도메인 특성상 **상하 반전 증강은 금지**했습니다.

---

## Slide 6 · 모델 선정 (0:45)

**제목:** Task에 맞는 모델 — YOLO11

**Bullet**
- **과제:** 객체 탐지 (Bounding Box)
- **선정:** Ultralytics **YOLO11** (표준·MPS 지원·실시간)

**후보 비교 (실측)**

| 모델 | mAP50 | 특징 |
|------|-------|------|
| YOLO11n (Baseline) | **0.538** | 경량·엣지 |
| YOLO11s (최종) | **0.575** | **+3.7%p** · 채택 |

**발표 멘트**
> 객체 탐지 표준인 YOLO11을 썼고, **Nano vs Small을 실제 학습·비교**해 Small을 채택했습니다.

---

## Slide 7 · 학습 설계 & 하이퍼파라미터 (1:00)

**제목:** 학습 설정 — 도메인 맞춤 + 튜닝

**표 (`configs/train.yaml`)**

| 항목 | 값 | 근거 |
|------|-----|------|
| Epoch | 50 | 수렴 확인 |
| Batch | 8 | M1 16GB OOM 방지 |
| imgsz | 640 | 속도·메모리 균형 |
| patience | 10 | Early stopping |
| LR | Cosine + warmup | 안정 수렴 |
| device | **mps** | M1 GPU |

**증강 (EXP 2)**
- HSV · Mosaic · Mixup · Erasing · **flipud=0**

**발표 멘트**
> 하이퍼파라미터는 YAML로 관리했고, batch 32는 **스왑**이 나와 **8**로 고정했습니다.

---

## Slide 8 · 실험 로그 EXP 1~3 (1:00)

**제목:** 성능 개선 — Baseline → 최종 (3단계 설계)

**표**

| 단계 | 변경 | mAP50 | 비고 |
|------|------|-------|------|
| **Baseline** | YOLO11n + min aug | **0.538** | 실측 |
| EXP1 | Nano→**Small** | (설계) | +스케일 |
| EXP2 | **도메인 증강** | (설계) | HSV·Mosaic |
| EXP3 | Epoch/Batch 튜닝 | (설계) | 50ep·batch8 |
| **최종** | 통합 | **0.575** | **+3.7%p** |

**한계**
- EXP별 독립 ablation 일부 미수행
- **EXP1 20ep** (`exp1_small_minaug`) 밤새 학습 진행 가능 — 결과 있으면 발표에 1줄 추가

**캡처 제안:** `runs/detect/train/results.png`

**발표 멘트**
> Baseline 대비 **3.7%p** 개선이 핵심 근거이고, EXP1~3은 **누적 설계**로 문서화했습니다.

---

## Slide 9 · 과적합 방지 (0:45)

**제목:** 일반화 — Regularization

**적용 ✅**
- Data Augmentation · Early stopping · Cosine LR · L2 (weight_decay)
- Pretrained YOLO11 · `close_mosaic: 10`

**미적용 (의도)**
- Dropout · L1 — YOLO 탐지 표준 관행

**캡처 제안:** Train/Val Loss 함께 수렴 (`results.png`)

**발표 멘트**
> Val Loss가 Train과 함께 수렴해 **심각한 과적합은 관찰되지 않았**습니다.

---

## Slide 10 · 최종 성능 (1:00)

**제목:** Validation 최종 성능

**표 (Val — `val_final`)**

| 지표 | 값 |
|------|-----|
| **mAP50** | **0.574** |
| mAP50-95 | 0.318 |
| **Precision** | **0.597** |
| **Recall** | **0.640** |

**Baseline vs 최종**
- 0.538 → **0.575** (train best) / **0.574** (val 재검증) = **+3.7%p**

**주의**
> 모든 수치는 **Validation set** 기준 (Test 미구축)

**캡처 제안:** `confusion_matrix.png`, `BoxF1_curve.png`

**발표 멘트**
> Val 기준 mAP50 **0.574**, Recall **0.64**입니다. Test가 없어 **일반화 성능은 보수적으로** 해석해야 합니다.

---

## Slide 11 · 클래스별 · FP/FN (1:00)

**제목:** 오류 분석 — 어디가 약한가?

**클래스별**

| 클래스 | P | R | mAP50 |
|--------|------|------|-------|
| Dirt | 0.521 | **0.750** | 0.549 |
| Damage | **0.673** | 0.530 | 0.599 |

**혼동행렬 핵심**

| 패턴 | 건수 |
|------|------|
| **Damage → Background (FN)** | **866** ← 핵심 |
| Background → Damage (FP) | 323 |
| Dirt ↔ Damage | **7** (낮음) |

**캡처 제안:** `runs/detect/val_final/val_batch0_pred.jpg`

**발표 멘트**
> 클래스 혼동은 적지만 **Damage 미탐**이 866건으로 가장 큰 이슈입니다. **소형 Damage**와 연결됩니다.

---

## Slide 12 · 추론 파이프라인 (0:45)

**제목:** Phase 1 Test — predict.py

**집계 (Val 2,694장)**

| 항목 | 값 |
|------|-----|
| 처리 이미지 | 2,694 |
| 탐지 있는 이미지 | 505 (18.7%) |
| 총 BBox | 1,294 |

**역할**
- `best.pt` 일괄 추론 · JSON 산출
- 공식 `val.py` 평가와 별도 **파이프라인 검증**

**발표 멘트**
> 학습 후 **Val 전체**에 추론 파이프라인을 검증했고, JSON으로 결과를 저장합니다.

---

## Slide 13 · Live Demo 소개 (0:30)

**제목:** 🎬 Live Demo — Streamlit 웹 데모

**전환 멘트**
> 이제 **코드 없이** 이미지를 올려 탐지하는 **웹 데모**를 보여드리겠습니다.

**실행**
```bash
python3 -m streamlit run app.py
```
→ http://localhost:8501

*(슬라이드는 최소 텍스트 — 데모용)*

---

## Slide 14 · Live Demo 시연 (2:00)

**시연 스크립트**

1. **업로드** — `DJI_0121_07_05.jpg` (또는 Val 이미지)
2. Confidence **0.25** 확인
3. **「결함 탐지 실행」** 클릭
4. **좌 원본 / 우 BBox** + Dirt·Damage **카운트** 설명
5. (15초) Confidence **0.15**로 낮춰 “Recall↑ · FP↑” 한 줄

**발표 멘트**
> 업로드만으로 **Dirt·Damage**가 표시됩니다. 임계값을 조절하면 **미탐 vs 오탐** trade-off를 현장에서 조정할 수 있습니다. 이게 **「탐지 → 서비스/데모」** 로의 확장입니다.

**데모 실패 대비:** Slide 11 예측 이미지 캡처 백업

---

## Slide 15 · Phase 2 API + 제품화 (1:15)

**제목:** 서비스 확장 — API & 제품화 로드맵

**Phase 2 ✅**

```
POST /api/v1/predict  →  JSON (BBox · class · summary)
Swagger: http://localhost:8000/docs
```

**실행**
```bash
python3 -m uvicorn backend.main:app --reload --port 8000
```

**제품화 로드맵**

| 단계 | 내용 | 상태 |
|------|------|------|
| **MVP** | FastAPI + Streamlit | ✅ |
| 단기 | Next.js 대시보드 · 검사 이력 · PDF 리포트 | 🔜 |
| 중기 | 드론 **엣지** 실시간 추론 · GPS 매핑 | 🔜 |
| 장기 | **검수자 피드백** · LogPick형 **예방 정비 SaaS** | 🔜 |

**발표 멘트**
> 데모 UI 뒤에는 **REST API**가 있어 B2B·모바일·드론과 연동 가능합니다. LogPick형 **예방 정비 SaaS**로 단계 확장할 계획입니다.

---

## Slide 16 · 심화 목표 (도전) (1:15)

**제목:** 심화 목표 (도전)

### ① 제품화를 한다면?
- **완료:** FastAPI + Streamlit MVP
- **로드맵:** Next.js · 검사 이력 · 엣지 추론 · 검수 피드백 SaaS

### ② 다른 모델 성능은?

| | 내용 |
|---|------|
| **완료** | YOLO11n vs s **실측** (+3.7%p) |
| **향후** | m/l · imgsz 1024 · **SAHI** (소형 Damage) |

### ③ 추가로 데이터를 수집?

| 이슈 | 수집 방향 |
|------|-----------|
| 극소형 Damage 93% | 고해상도·근접 촬영 |
| Dirt 부족 (15:1) | Dirt 라벨 보강 |
| FN 866 | 안개·역광 도메인 |
| Test 없음 | **Hold-out Test** 구축 |
| 영상 10초 | **라벨링 후** 단계 반영 (단순 추가 X) |

**발표 멘트**
> 모델·데이터 모두 **EDA와 오류 분석에 근거**한 로드맵입니다.

---

## Slide 17 · 한계 & 향후 (0:45)

**제목:** 한계 & Lessons Learned

**한계**
- Test 세트 없음 → Val 기준 평가
- Damage **미탐(FN)** 다수 (소형 객체)
- EXP 독립 ablation 일부 미완

**강점**
- **13K** 데이터 · End-to-End 파이프라인
- Baseline **정량 비교** · Notion 자동 리포트
- **웹 데모 + API** (평가: 서비스/데모 시도)

**발표 멘트**
> 완벽한 Test 성능보다 **파이프라인 완주 + 데모 + 분석**에 무게를 뒀습니다.

---

## Slide 18 · 마무리 (0:45)

**제목:** Thank You

**한 줄 요약**
> **YOLO11s** · Val mAP50 **0.574** · **Streamlit Live Demo** · **예방 정비 SaaS** 로드맵

**링크**
- GitHub: https://github.com/toryhyeon80/wind-turbine-yolo
- Notion 리포트
- 데모: `localhost:8501` (발표 시)

**Q&A**

**발표 멘트**
> 풍력 블레이드 **Dirt/Damage 자동 탐지**부터 **웹 데모**까지 구현했습니다. 질문 받겠습니다.

---

## 백업 Slide B1 · 평가 항목 매핑

| 평가 항목 | 우리 슬라이드 |
|-----------|----------------|
| EDA·전처리 | 4~5 |
| 적절한 모델 | 6 |
| 논리적 성능 향상 | 7~8 |
| 여러 시도 | 8~9 |
| 설득력 있는 결론 | 10~11 |
| Metric 분석 | 10~11 |
| **서비스/데모** | **13~15** |
| 발표 매끄러움 | Live Demo 리허설 |

---

## 백업 Slide B2 · EXP1 / 추가 그래프

- EXP1 `runs/detect/exp1_small_minaug` 결과 (있으면 mAP50 추가)
- Notion 리포트 스크린샷

---

## 캡처·그래프 체크리스트

| # | 자산 | 슬라이드 |
|---|------|----------|
| 1 | `runs/eda/class_distribution.png` | 5 |
| 2 | `runs/eda/bbox_size_distribution.png` | 5 |
| 3 | `runs/detect/train/results.png` | 8~9 |
| 4 | Baseline vs 최종 mAP50 막대 (0.538 vs 0.574) | 6, 10 |
| 5 | `runs/detect/val_final/confusion_matrix.png` | 11 |
| 6 | `runs/detect/val_final/val_batch0_pred.jpg` | 11 |
| 7 | **Streamlit 탐지 완료** 화면 캡처 | 13~14 |
| 8 | FastAPI Swagger `/docs` (선택) | 15 |
| 9 | 파이프라인 다이어그램 | 3 |

---

## Gamma AI 사용 팁

1. **입력:** 이 파일 전체 또는 `## Slide N` 단위로 붙여넣기
2. **테마:** B2B·테크 — Navy `#1E3A5F` · Teal `#0D9488` (`DESIGN.md` LogPick)
3. **슬라이드 수:** 18~20장 목표
4. **이미지:** 위 체크리스트 경로에서 캡처 후 Gamma에 수동 삽입
5. **Live Demo 슬라이드(13~14):** 텍스트 최소 · 큰 캡처 1장

---

## 발표 리허설 (15분)

1. **Live Demo 2분** — 실패 시 Slide 11 캡처로 대체
2. **숫자 3개:** 13,470장 · mAP50 **0.574** · **+3.7%p**
3. **한계 1문장:** Val 기준, Test·Damage FN은 향후 과제
4. 발표 전 실행:
   - `python3 -m streamlit run app.py`
   - (선택) `python3 -m uvicorn backend.main:app --port 8000`

---

## EXP1 밤새 학습 (선택)

```bash
cd ~/wind-turbine-yolo
nohup python3 train.py --config configs/train_exp1_small_minaug.yaml --no-report > exp1_train.log 2>&1 &
```

- 설정: `configs/train_exp1_small_minaug.yaml`
- 산출: `runs/detect/exp1_small_minaug/`
- 아침 결과 있으면 Slide 8·B2에 mAP50 1줄 추가
