# 🌪️ Wind-Turbine AI — 오픈소스·문서 참조 (OSS Sources)

> **역할**: 풍력 발전기 파손 탐지 해커톤의 **탐색 지도 및 기술 스택 명세서**.  
> **에이전트 지침**: 코드 구현 시 이 문서에 채택된 OSS(오픈소스)만 엄격하게 참조하며, 각 Phase(단계)에 맞지 않는 기술은 미리 구현하지 마십시오.

---

## 1. 채택 OSS 요약 (해커톤 Phase 매핑)

| ID  | 기술                   | 용도                               | 해커톤 Phase          | 공식 문서/참조                                          |
| --- | ---------------------- | ---------------------------------- | --------------------- | ------------------------------------------------------- |
| S1  | **Ultralytics YOLO**   | 객체 탐지 (YOLO11)                 | **Phase 1** (Train)   | [docs.ultralytics.com](https://docs.ultralytics.com/)   |
| S2  | **OpenCV (`cv2`)**     | 추론 결과 시각화 및 이미지 전처리  | **Phase 1** (Test)    | [docs.opencv.org](https://docs.opencv.org/)             |
| S3  | **FastAPI**            | AI 추론 결과를 반환하는 백엔드 API | **Phase 2** (Backend) | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)   |
| S4  | **Next.js + Tailwind** | B2B SaaS 데모용 프론트엔드         | **Phase 3** (Web UI)  | `DESIGN.md` 참조                                        |
| S5  | **notion-client**      | mAP 결과 및 로그 자동화 리포팅     | 공통 (Logging)        | [developers.notion.com](https://developers.notion.com/) |

---

## 2. 영역별 상세 및 독학/구현 순서

### Phase 1: AI 엔진 학습 및 평가 (현재 단계)

- **목표:** `split_data.py`로 데이터를 나누고, `train.py`로 YOLO 모델 학습.
- **제약:** Mac 16GB 램을 고려하여 모델은 `yolo11s.pt` (Small 버전)만 사용.
- **증강(Augmentation):** `flipud=0.0`(상하 반전 금지) 등 도메인 지식 필수 적용.

### Phase 2: 백엔드 API 서버 (`backend/`)

- **목표:** 학습된 `best.pt` 가중치를 로드하고, 프론트엔드에서 이미지를 받으면 Bounding Box와 클래스(Dirt/Damage)를 JSON으로 반환.
- **주의:** Phase 1의 학습이 완전히 끝나서 가중치 파일이 생성된 이후에 착수.

### Phase 3: 프론트엔드 B2B 웹 데모 (`frontend-user/`)

- **목표:** `DESIGN.md`의 LogPick 브랜드 테마(Navy/Teal)를 적용한 결과 시각화 대시보드.
- **UI 구조:** 좌측(업로드 및 원본) / 우측(YOLO 추론 결과 및 발견된 객체 수 카운팅).

---

## 3. 에이전트 CLI 사용법 (프롬프트 템플릿)

에이전트에게 명령을 내릴 때 아래의 텍스트 블록을 활용하십시오.

**[Phase 1] 학습 스크립트 작성 시:**

```text
@CLAUDE.md @references.md
Phase 1 규칙에 따라 데이터를 8:2로 나누는 `split_data.py`와 YOLOv11 학습을 시작하는 `train.py`를 작성해 줘.
(Apple M1 GPU `mps` 세팅 필수)
```
