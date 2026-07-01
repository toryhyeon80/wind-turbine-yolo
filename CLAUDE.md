# 🛠️ AI 에이전트 개발 행동 강령 (YOLO 해커톤 통합 에디션)

> **[필독] 에이전트 구동 지침**
>
> 1. 본 프로젝트는 '풍력 발전기 파손 탐지 YOLO AI'와 이를 서비스하는 'B2B 웹 프로덕트'를 통합 구축합니다.
> 2. **AI/ML 작업:** Apple M1 Pro(`device='mps'`)를 활용하며, 코드 수정 후 반드시 1 Epoch 테스트로 검증하십시오.
> 3. **UI/UX 작업:** 모든 프론트엔드 작업은 루트 디렉토리의 `DESIGN.md` 문서를 절대적인 기준으로 삼아 디자인 토큰, 색상(LogPick Navy/Teal), 폰트(Pretendard)를 엄격히 적용하십시오.

---

## 1. 파이프라인 및 테스트 명령어 (Commands)

에이전트는 각 파트의 코드를 수정한 후 반드시 아래의 검증(Test) 명령어를 실행하십시오.

### ① AI 머신러닝 파이프라인 (YOLOv11)

- **데이터 전처리:** `python split_data.py` (8:2 Train/Val 분할 및 라벨 매칭 검증)
- **모델 학습:** `python train.py` (mps 가속 및 Augmentation 적용 상태 확인)
- **결과 검증:** `runs/detect/train/` 폴더 내 결과물 확인 및 노션 자동화 리포트(`update_notion.py`) 업데이트.

### ② 백엔드 (Backend - FastAPI)

- **역할:** 학습된 YOLO 가중치(`best.pt`)를 로드하여 프론트엔드에서 보낸 이미지를 분석하고 JSON(BBox 좌표, 클래스)으로 반환.
- **실행 명령어:** `uvicorn backend.main:app --reload`
- **검증:** Postman 또는 Swagger UI(`http://localhost:8000/docs`)를 통한 이미지 업로드 테스트.

### ③ 프론트엔드 (Frontend - Next.js)

- **역할:** `DESIGN.md`를 준수한 사용자 웹(데모 시연용) 구축.
- **실행 명령어:** `npm install` 후 `npm run dev`
- **검증:** API 서버 통신 상태 및 반응형(Responsive) 레이아웃 렌더링 확인.

### ④ 중간 결과 시각화 및 로깅 (Visualization & Logging)

- **중간 산출물 보존:** 학습 및 검증 과정에서 도출되는 모든 시각적 결과물(Loss/mAP 그래프, 오차 행렬, BBox 예측 예시 이미지 등)은 해커톤 발표 자료의 핵심 근거입니다.
- **에이전트 역할 (보고서 자동화):**
  1. 새로운 실험(1개 Epoch 세트 완료 또는 파라미터 튜닝)이 끝날 때마다 `runs/detect/` 최신 폴더를 스캔하십시오.
  2. 스캔한 주요 이미지(`results.png`, `val_batch0_pred.jpg` 등)를 `report.md` 문서의 해당 실험 항목에 마크다운 이미지 링크(`![설명](상대경로)`) 형태로 즉각 삽입하십시오.
  3. 사용자가 텍스트뿐만 아니라 시각적인 그래프와 예측 이미지를 통해 모델의 개선 과정을 한눈에 추적할 수 있도록 문서를 구성해야 합니다.

---

## 2. 프론트엔드 UI/UX 설계 원칙 (Based on DESIGN.md)

커서 AI는 컴포넌트 생성 시 다음 규칙을 강제합니다.

1. **디자인 테마:** LogPick Core의 '차분하고 밀도 있는 B2B SaaS' 무드를 유지합니다. 과도한 그라데이션, 화려한 장식은 배제합니다.
2. **컬러 앤 타이포그래피:**
   - **Primary Color:** LogPick Teal (`#0D9488`), 배경/헤더: LogPick Navy (`#1E3A5F`)
   - **폰트:** 1순위 `Pretendard` 적용.
3. **컴포넌트 설계:**
   - Tailwind CSS를 활용하며, `DESIGN.md` §4에 명시된 버튼(Primary, Secondary 등) 및 카드(`rounded-xl`) 규격을 준수합니다.
   - 데모 화면은 좌측 '원본 이미지 업로드', 우측 'YOLO 탐지 결과 및 수치 표기'의 2단 레이아웃을 기본으로 합니다.

---

## 3. 핵심 코딩 스타일 가이드라인 (Coding Rules)

### ① 도메인 맞춤형 AI 로직

- 풍력 발전기 도메인 특성상 상하 반전(`flipud`)은 절대 금지(`0.0`)하며, 팀에서 지정한 최적의 하이퍼파라미터만 사용합니다.
- 모든 환경 변수(포트 번호, 모델 경로, API 토큰)는 `.env` 파일로 분리합니다.

### ② 에러 핸들링

- **AI:** `OOM` 발생 시 배치 사이즈 축소. 경로 에러 시 `data.yaml` 점검.
- **Web:** 백엔드/프론트엔드 간 CORS 에러 방지 처리 및 API 타임아웃 예외 처리 필수. 에러 발생 시 전체 코드를 갈아엎지 않고 해당 로직만 핀포인트로 디버깅합니다.

---

## 4. Git 커밋·푸시 규칙 (Version Control)

- **[커밋 제외 `.gitignore`]** 데이터셋(`data/`), 모델 가중치(`*.pt`), 환경변수(`.env`), `node_modules/`
- **[커밋 타이밍]** 각 기능 단위 완료 시 (예: `feat(backend): YOLOv11 추론 엔드포인트 구현`, `design(frontend): 탐지 결과 카드 UI 적용`)
