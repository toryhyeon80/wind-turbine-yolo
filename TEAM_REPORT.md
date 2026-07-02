# 🌪️ 팀 기획서 vs 실제 프로젝트 대조 보고서 (TEAM_REPORT)

> **용도:** 팀원 공유 · 발표 멘트 · PPT/리포트 문구 정합성 검토  
> **기준일:** 2026-07-02  
> **대상:** 팀 기획 문서(§1~§11) ↔ 레포 실제 구현 (`report.md`, `CLAUDE.md`, 코드베이스)  
> **관련 문서:** `report.md`(Phase 1 상세), `PRESENTATION.md`(발표), `TRAINING_CHECKLIST.md`(체크리스트)

---

## 0. 문서 목적

팀에서 작성한 기획·조사 문서(문제 정의, YOLO 선정, 데이터 전략, 평가 지표, 산업 적용 등)는 **이론·계획** 중심이다.  
본 레포는 **실제 구현 + 일부 계획 미반영** 상태이다. 발표·심사에서 **과장 없이** 말하기 위해 섹션별로 대조하고, **우리 프로젝트에 맞는 표현**을 정리한다.

### 상태 범례

| 표기 | 의미 |
| :--- | :--- |
| ✅ | 기획과 구현 일치 (또는 실측 완료) |
| 🟡 | 부분 반영 / 수치·범위 조정 필요 |
| ❌ | 미구현 (향후 과제) |
| 📝 | 문서만 있음 (문헌·계획, 실험 없음) |

---

## 1. 한눈에 보는 대조표

| 문서 섹션 | 기획서에서 말하는 것 | 우리 프로젝트 실제 | 일치도 |
| :--- | :--- | :--- | :---: |
| §1 자료 조사·문제 정의 | 드론 이미지 → BBox 탐지, 분류 아님 | Dirt/Damage 2클래스 YOLO11 탐지 | ✅ |
| §2 Task 요구사항 | 실시간·위치·소형객체·산업 확장 | BBox·파이프라인·데모까지. **FPS·DB·이력관리 없음** | 🟡 |
| §3 YOLO 선정 근거 | YOLOv8, Ultralytics | **YOLO11**, Ultralytics, MPS | 🟡 |
| §4 타 모델 비교 | Faster R-CNN·RetinaNet·RT-DETR | **문헌 근거만**, 벤치마크 실험 없음 | 📝 |
| §5 모델 크기 비교 | n / s / **m** + FPS | **n→s 완료**, EXP1 진행, **m·FPS 없음** | 🟡 |
| §6 성능 향상 접근 | 증강·다양성·모델 비교·메트릭 | `configs/train.yaml` EXP1~3 + Baseline 비교 | ✅ |
| §7 데이터 수집 | AI Hub·Roboflow·직접촬영 | **13,470장 구축**, 외부 소스 **미연동** | 🟡 |
| §8 평가 지표 근거 | Recall 최우선 | 전체 R **0.640**, Damage R **0.530** → **갭 존재** | 🟡 |
| §9 메트릭 분석법 | P/R/F1/mAP 시나리오별 대응 | 혼동행렬·FN 866건 **실측 반영** | ✅ |
| §10 산업 적용 | DB·대시보드·위험도·예측정비 | **Streamlit + FastAPI MVP**, 심화는 로드맵 | 🟡 |
| §11 최종 결론 | v8n/s/m, 다양한 데이터 소스 | **§11 수정본** 참고 | 📝 |

---

## 2. 프로젝트 스냅샷 (실측 기준)

### Phase 진행

| Phase | 내용 | 상태 |
| :--- | :--- | :---: |
| Phase 1 | 분할 · EDA · 학습 · 검증 · 리포트 · Notion | 🟢 완료 |
| Phase 1 Test | `predict.py` Val 일괄 추론 · JSON | 🟢 완료 |
| Phase 2 | FastAPI `POST /api/v1/predict` | 🟢 MVP |
| Phase 3 | Streamlit Live Demo (`app.py`) | 🟢 MVP |

### 핵심 수치 (Validation · `val_final`)

| 구분 | 모델 | mAP50 | mAP50-95 | 비고 |
| :--- | :--- | ---: | ---: | :--- |
| Baseline | YOLO11n | 0.538 | 0.317 | `configs/train_baseline.yaml` |
| 최종 | YOLO11s | 0.575 | 0.319 | `configs/train.yaml` |
| 재검증 | YOLO11s | 0.574 | 0.318 | `runs/detect/val_final/` |

- **데이터:** Train 10,776 / Val 2,694 (**총 13,470장**, 8:2) · 라벨 2,995 / 배경 10,475  
- **클래스:** `dirt`(0), `damage`(1)  
- **환경:** Apple M1 Pro · `device=mps` · `imgsz=640` · batch 8  
- **Test 세트:** ❌ 없음 → Val 기준 평가  
- **GitHub:** https://github.com/toryhyeon80/wind-turbine-yolo  
- **리포트 자동화:** `report.md` · Notion · `report/assets/` (GitHub 이미지 공개)

---

## 3. 섹션별 상세 대조

### §1~2 · 문제 정의 & Task 요구사항

**기획서 요지:** 풍력 블레이드 표면 손상·오염, 육안 검수 한계 → **객체 탐지(BBox)** 가 적합.  
요구 4가지: **실시간성**, **위치 탐지**, **소형 객체**, **산업 확장성**.

| 요구사항 | 우리 반영 | 근거 |
| :--- | :---: | :--- |
| 위치 탐지 (BBox) | ✅ | YOLO, `predict.py` JSON, Streamlit 시각화 |
| 소형 객체 | 🟡 | EDA: Damage BBox **93% 극소형** · FN **866건** |
| 실시간성 | ❌ | FPS 벤치마크 없음 (MPS 학습·추론만) |
| 산업 확장 | 🟡 | FastAPI + Streamlit MVP |

**발표용 문장**

> 실시간 드론 탑재는 **향후 Edge 배포 목표**로 두고, 해커톤에서는 Val **2,694장 일괄 추론 파이프라인**과 **웹 데모**로 산업 적용 가능성을 검증했다.

---

### §3~4 · YOLO 선정 & 타 모델 비교

**기획서:** YOLOv8 One-Stage, 속도·정확도 균형. Faster R-CNN(느림), RetinaNet(배포 불편), RT-DETR(무거움).

**우리 프로젝트**

- 채택: **YOLO11** (기획서의 “v8” → 발표·문서에서 **YOLO11**로 통일)
- 프레임워크: **Ultralytics** · Train/Val/Predict/리포트 자동화
- 타 모델: **선정 근거(문헌)** 만 — 동일 데이터로 Faster R-CNN 등 **실험 비교 없음**

**발표용 문장**

> 드론·대량 이미지 환경에서 **추론 속도와 Ultralytics 파이프라인**을 고려해 YOLO 계열을 채택했으며, Two-Stage·Transformer 계열은 정밀도는 높으나 **현장 배포 부담**이 커 제외했다.

---

### §5 · 모델 크기별 비교 (n / s / m)

**기획서:** n=Baseline, s=서비스 후보, m=고정밀, Precision·Recall·F1·mAP·**FPS** 비교.

**우리 실측**

| 모델 | Epoch | mAP50 | mAP50-95 | 설정 |
| :--- | ---: | ---: | ---: | :--- |
| YOLO11n | 18 | 0.538 | 0.317 | `train_baseline.yaml` |
| YOLO11s (최종) | 50 | 0.575 | 0.319 | `train.yaml` |
| YOLO11s (EXP1) | 20 | 진행 중 | — | `train_exp1_small_minaug.yaml` |

- **YOLO11m:** ❌ 미학습 — “계획”이지 “결과” 아님  
- **FPS:** ❌ 미측정

**발표용 문장**

> Nano로 파이프라인을 검증한 뒤 Small로 스케일업해 mAP50 **+3.7%p**를 얻었고, Medium·FPS 비교는 **일정상 후속 과제**로 남겼다.

---

### §6 · 성능 향상 논리 & 데이터 증강

**기획서:** 데이터 다양성 → 증강(반전, 밝기·대비·채도, Blur·Noise) → 모델 크기 → 메트릭 분석.

**우리 EXP 설계 (`report.md` §3)**

| 단계 | 변경 | 구현 |
| :--- | :--- | :--- |
| **EXP 1** | Nano → Small | 모델 스케일업 (+ EXP1 독립 ablation 진행) |
| **EXP 2** | HSV·Mosaic·Mixup·Erasing | `flipud=0` 도메인 규칙 포함 |
| **EXP 3** | 50ep · batch 8 · patience 10 | M1 16GB OOM 대응 · Cosine LR |

**차이점**

- Blur·Noise: 기획서에 있으나 YAML **명시 항목 없음** (HSV·erasing으로 부분 대응)
- EXP1~3: **누적 설계** — Baseline↔최종 정량 비교는 있으나, **단계별 독립 ablation 전부 완료는 아님**

---

### §7 · 데이터 수집 전략

**기획서 우선순위:** ① 증강 ② Roboflow ③ AI Hub ④ 직접 촬영(장기) ⑤ 유사 도메인 pretrain

| 방법 | 기획서 | 우리 프로젝트 |
| :--- | :--- | :--- |
| 보유 데이터셋 | Kaggle 등 | ✅ **13,470장** (드론 블레이드) |
| Data Augmentation | 1순위 | ✅ `configs/train.yaml` |
| Roboflow Universe | 2순위 | ❌ 미적용 |
| AI Hub 유사 도메인 | 3순위 | ❌ 미적용 |
| 직접 촬영 | 장기 | ❌ 로드맵 |
| 유사 도메인 pretrain → fine-tune | §7.4 | ❌ 미수행 |

**발표용 문장**

> 당장은 **보유 드론 이미지 + 도메인 증강**으로 일반화를 확보했고, AI Hub·Roboflow·현장 촬영은 **데이터 확장 로드맵**이다.

---

### §8~9 · 평가 지표 & 결과 해석

**기획서:** Recall 최우선(안전), Precision(운영), F1(균형), mAP50/95, 혼동행렬 시나리오별 대응.

**우리 Val 실측 (`val_final`)**

| 지표 | 전체 | Dirt | Damage |
| :--- | ---: | ---: | ---: |
| Precision | 0.597 | 0.521 | 0.673 |
| Recall | 0.640 | 0.750 | **0.530** |
| mAP50 | 0.574 | 0.549 | 0.599 |

**혼동행렬 핵심 (BBox 단위)**

| 패턴 | 건수 | 해석 |
| :--- | ---: | :--- |
| **Damage → Background (FN)** | **866** | Damage 미탐 — **핵심 이슈** |
| Background → Damage (FP) | 323 | 배경 오탐 |
| Dirt ↔ Damage 혼동 | 7 | 클래스 간 혼동 낮음 |

**기획서 시나리오 ↔ 우리 상태**

- “Recall 최우선”이나 **Damage Recall 0.530** — 극소형 Damage에서 미탐 집중
- mAP50 **0.574** vs mAP50-95 **0.318** — BBox **위치 정밀도** 개선 여지

**발표용 문장**

> 안전 관점에서 Recall을 최우선으로 설계했으나, **극소형 Damage**에서 미탐이 집중되어 Damage Recall **0.530**이 병목이다. 향후 conf 조정·소형 객체 증강·해상도 상향으로 보완한다.

---

### §10 · 산업 적용 & 제품화

**기획서:** 드론 → AI → **DB 이력** → 위험도 → **관리자 대시보드** → 멀티모달 예측정비.

| 기능 | 상태 | 경로 |
| :--- | :---: | :--- |
| 이미지 업로드 → BBox 탐지 | ✅ | `app.py` (Streamlit) |
| REST 추론 API | ✅ | `backend/`, `configs/api.yaml` |
| 결함 이력 DB | ❌ | 향후 |
| 위험도·유지보수 우선순위 | ❌ | PPT 로드맵 |
| LogPick 풀 UI (Navy/Teal) | ❌ | `DESIGN.md` 확장 예정 |

**발표용 문장**

> Phase 1은 **탐지 엔진 + 리포트 자동화**, Phase 2·3은 **API·웹 데모 MVP**까지 완료했고, 이력·위험도·예측정비는 **B2B SaaS 로드맵**이다.

---

## 4. §11 최종 결론 — 프로젝트 반영본

아래를 팀 발표·심사 **결론 슬라이드**에 사용한다.

---

본 프로젝트는 드론으로 촬영한 풍력 터빈 블레이드 이미지에서 **Dirt(오염)** 와 **Damage(손상)** 을 **YOLO11 객체 탐지**로 자동 탐지하는 **스마트 예방 정비 시스템의 Phase 1**을 구축했다.

**기술 선정:** 대량 이미지·현장 배포를 고려해 **YOLO(One-Stage)** 를 채택했고, Two-Stage·Transformer 계열은 속도·배포 부담으로 제외했다. **Ultralytics + Apple MPS**로 학습·검증·추론·리포트까지 End-to-End 파이프라인을 자동화했다.

**데이터:** 총 **13,470장**(Train 10,776 / Val 2,694), 배경 이미지 포함. Test 세트는 없고 **Val 기준** 평가. 데이터 다양성은 **도메인 증강**(HSV, Mosaic, Mixup, `flipud=0`)으로 보완했다.

**모델:** **YOLO11n Baseline**(mAP50 0.538) 대비 **YOLO11s 최종**(mAP50 **0.574~0.575**, Precision 0.597, Recall 0.640). Small 스케일업으로 **+3.7%p** 개선. YOLO11m·FPS 비교·AI Hub/Roboflow 연동은 후속 과제다.

**평가:** mAP50, mAP50-95, P/R, 혼동행렬, FP/FN 사례를 `val.py`·`predict.py`·`report.md`에 자동 반영했다. **Damage 미탐(FN 866)** 이 핵심 개선 포인트이며, EDA상 Damage BBox **93%가 극소형**임을 확인했다.

**서비스:** **FastAPI** 추론 API와 **Streamlit** Live Demo까지 MVP를 완료했고, 결함 이력·위험도 분석·예측 정비는 향후 LogPick B2B 대시보드로 확장할 계획이다.

**한계:** 독립 Test 세트 없음, EXP 단계별 완전 ablation 미완, Edge FPS 미측정, Damage Recall 보완 필요.

---

## 5. 4인 역할 분담 (문서 섹션 매핑)

| 팀원 | 담당 기획 섹션 | 실제 할 일 | 발표 구간 (`PRESENTATION.md`) |
| :--- | :--- | :--- | :--- |
| **A · ML/실험** | §5, §6, §9 | EXP1 결과, Damage FN 분석, conf/증강 개선안 | Slide 7~12 |
| **B · 데이터/EDA** | §1, §7, §8 | 13,470장·불균형·극소형 Damage 스토리, 데이터 로드맵 | Slide 4~6 |
| **C · 제품/데모** | §2, §10 | Streamlit·FastAPI 데모, 산업 확장 로드맵 | Slide 13~15 |
| **D · PM/문서** | §3, §4, §11 | YOLO11 선정·문헌 비교, 결론, Notion·GitHub·PPT | Slide 1~3, 16~18 |

### 협업 규칙

1. **설정은 YAML만** — `configs/*.yaml`, `data/data.yaml` (A·B)
2. **리포트는 자동화** — `python3 update_report.py` → `report.md` / `report/assets/` (D)
3. **Notion** — `python3 update_notion.py` (D, 학습 종료 후)
4. **데모** — `python3 -m streamlit run app.py` (C)
5. **EXP 학습** — `--no-report`로 밤새 돌릴 때 Notion 메트릭 혼선 방지 (A)

---

## 6. 발표 전 표현 정리 (오류 방지)

| 기획서·습관적 표현 | 우리 프로젝트에서 쓸 표현 |
| :--- | :--- |
| YOLOv8n / s / m | **YOLO11n / YOLO11s** (m은 **계획**) |
| Recall 최우선 **달성** | 전체 R 0.640, **Damage R 0.530** — 목표와 **갭 인정** |
| n/s/m + FPS 실험 **완료** | **n vs s 실측**, m·FPS는 **향후** |
| AI Hub·Roboflow **활용** | **증강 + 보유 데이터** 중심, 외부 소스는 **로드맵** |
| 관리자 대시보드 **구축** | **Streamlit MVP** + API, DB/이력은 **로드맵** |
| Test 세트로 최종 평가 | **Val 기준** (Test 미구축 — 정책 명시) |
| EXP1~3 각각 독립 실험 완료 | **누적 설계** + Baseline↔최종 비교 (ablation 일부 미완) |

---

## 7. 향후 과제 (기획서 → 로드맵)

| 우선순위 | 항목 | 담당 제안 |
| :---: | :--- | :--- |
| 1 | EXP1 결과 반영 · Small+최소증강 vs Baseline 표 | A |
| 2 | Damage FN 개선 (conf, 소형 객체 증강, imgsz) | A + B |
| 3 | FPS / Edge 추론 벤치마크 (M1 또는 Jetson) | A + C |
| 4 | YOLO11m 비교 실험 (시간 허용 시) | A |
| 5 | AI Hub / Roboflow 데이터 검토·클래스 매핑 | B |
| 6 | 결함 이력 DB + 위험도 스코어링 (Phase 4) | C + D |
| 7 | LogPick 풀 UI (`DESIGN.md`) | C |
| 8 | 독립 Test 세트 10~15% 확보 | B |

---

## 8. 관련 명령어 (팀 공통)

```bash
# 리포트·GitHub 이미지 갱신
python3 update_report.py
git add report.md report/assets/
git commit -m "docs: report 이미지·수치 갱신"

# Notion 동기화 (학습 완료 후)
python3 update_notion.py --skip-eda

# 데모
python3 -m uvicorn backend.main:app --reload --port 8000
python3 -m streamlit run app.py

# EXP1 (밤새, 리포트 갱신 생략)
nohup python3 train.py --config configs/train_exp1_small_minaug.yaml --no-report > exp1_train.log 2>&1 &
tail -f exp1_train.log
```

---

## 9. 문서 간 역할 분리

| 문서 | 역할 |
| :--- | :--- |
| **TEAM_REPORT.md** (본 문서) | 기획서 vs 실구현 대조 · 팀 역할 · 발표 문구 |
| **report.md** | Phase 1 기술 리포트 (자동 갱신) · Notion 원본 |
| **PRESENTATION.md** | 15분 발표 슬라이드·멘트 |
| **TRAINING_CHECKLIST.md** | 루브릭·체크리스트 증빙 |
| **CLAUDE.md** | 개발·에이전트 행동 강령 |

---

*최종 갱신: 2026-07-02 · EXP1 학습 진행 중일 수 있음 — 수치는 `report.md` auto 섹션을 최우선 참조.*
