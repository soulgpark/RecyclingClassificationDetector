# ♻️Recycling Classification Detector

이 프로젝트는 실시간 쓰레기 분류하고, 제스처 기반 드래그하는 기능을 제공합니다. 사용자는 자신의 웹캠을 통해 실제 쓰레기 사진을 보여주면, 시스템은 이를 자동으로 인식하고 해당 쓰레기의 종류를 화면에 표시합니다. 이후 인식된 쓰레기 이미지를 손가락으로 직접 드래그하여 분리수거 통 영역으로 이동시킬 수 있습니다.


https://github.com/user-attachments/assets/420aebdf-6b09-40df-b799-943b43be572a


---

## 모델 설명

### 1. YOLOv8 Classification

* **모델 아키텍처**: Ultralytics의 YOLOv8 분류 모델은 경량화된 CNN 기반의 분류 네트워크입니다. 기본 블록으로 CSPDarknet과 BottleneckCSP 구조를 사용하며, 최종 출력층은 클래스별 확률을 계산하는 Fully Connected 레이어로 구성됩니다.
* **사전 학습 모델**: `yolov8n-cls.pt` (nano) 모델 활용하여 쓰레기 데이터셋에 맞춰 미세 조정 가능합니다.
* **커스텀 학습**: 필요에 따라 `Ultralytics YOLO` API를 통해 커스텀 데이터셋으로 재학습하며, 학습 시 `epochs`, `batch size`, `learning rate`를 조절합니다.

### 2. 투표 기반 안정화(Voting)

*  실시간 분류 시 순간적인 오탐지를 줄이고, 사용자에게 안정적인 결과를 제공하기 위해 프레임 단위 예측을 모아서 다수결 방식으로 최종 클래스를 결정합니다.
* **방법**:

  1. 연속된 `VOTE_FRAMES`(30프레임)마다 모델에서 예측된 클래스 레이블을 리스트에 저장.
  2. 리스트의 최빈값을 구해 최종 클래스 결정.
  3. 선택된 클래스가 이전 클래스와 다를 경우 일정 시간 이상 유지되어야 변경 적용.

---

## 데이터셋 구성

1. **폴더 구조**:

   ```
   trash_dataset/
   ├─ train/
   │  ├─ plastic/
   │  ├─ paper/
   │  └─ can/
   └── val/
      ├── plastic/
      ├── paper/
      └── can/
   ```
2. **클래스별 이미지**: 각 클래스 폴더에는 해당 쓰레기 종류의 이미지 파일들을 저장합니다.
3. **전처리**:

   * **크기 통일**: 모든 이미지를 `imgsz`(256px)로 리사이즈.
   * **정규화**: 픽셀 값을 `[0,1]` 범위로 스케일링.
   * **데이터 증강(option)**: 회전, 플립, 밝기 변경 등을 적용해 모델 일반화 성능 향상.
4. **학습/검증 분할**: 80% 학습, 20% 검증 세트 구성.

---

## 전체 프로세스

다음 다이어그램은 시스템의 흐름을 요약한 것입니다

```
flowchart LR
    A[웹캠 프레임 캡처] --> B[분류 (YOLOv8 + Voting)]
    B --> C[결과 저장 & 상단 GUI 표시]
    C --> D[손 랜드마크 추출 (MediaPipe)]
    D --> E{아이콘 드래그 상태}
    E -->|드래그 시작| F[아이콘 위치 업데이트]
    F --> G{통 영역 드롭 감지}
    G -->|정확| H[애니메이션 이동]
    G -->|오분류| I[원위치 복귀 & 경고 메시지]
```

1. **웹캠 프레임 캡처**: OpenCV `VideoCapture`를 통해 연속된 프레임 획득.
2. **분류**: `vote_classification()` 함수에서 YOLOv8 모델 예측 및 Voting 처리.
3. **결과 저장**: 최종 클래스 확정 시 이미지(`current_trash.jpg`)와 텍스트(`trash_type.txt`)로 로컬 저장.
4. **GUI 표시**: Tkinter 캔버스 상단에 쓰레기 이미지와 재활용 안내문 표시.
5. **손 추적 & 드래그**: MediaPipe Hands로 인덱스 끝 좌표 추출, 드래그 로직 실행.
6. **드롭 처리**: 통 위치와 드래그 위치를 비교하여 정답 시 애니메이션 이동, 오답 시 원위치 복귀 및 경고.

---

## 주요 기능 상세

### 1. 실시간 분류

* **프레임 캡처**: OpenCV `VideoCapture`로 웹캠 프레임을 획득하고, 호출 주기를 일정하게 유지합니다.
* **분류 함수**:

  1. 프레임을 모델 입력 크기에 맞춰 전처리
  2. YOLO 모델로 추론
  3. 결과에서 확률이 가장 높은 클래스 추출
  4. 리스트에 저장 후 투표 방식으로 안정화

### 2. 이미지 안내 출력

* **상단 이미지**: 상단에 쓰레기 이미지를 나타냅니다.
* **안내 텍스트**: 재활용 방법을 `RECYCLING_INSTRUCTION` 딕셔너리에서 가져와 표시합니다.

### 3. 제스처 기반 드래그

* **손 추적**: MediaPipe Hands 모듈을 사용하여 손 랜드마크를 추출하고, 인덱스 끝(`landmark[8]`) 좌표를 기준으로 드래그 포인터를 생성합니다.
* **드래그 로직** (`drag_item` 함수):

  1. 인덱스 끝 좌표와 아이콘 영역이 겹치면 `dragging=True`.
  2. `dragging` 상태에서 손 움직임에 따라 아이콘 위치 업데이트.
  3. 목표 통(`bins`) 영역에 닿으면 `on_drop` 호출.
* **드롭 처리** (`on_drop` 함수):

  * 정확한 통: `animate_to_bin` 함수를 통해 스무스 애니메이션.
  * 오분류: 원위치 복귀와 함께 `show_warning` 함수로 메시지 표시.

### 4. GUI 시각화

* **GUI 메시지**: 상태 표시줄에 `정확히 분류되었습니다!` 또는 `잘못된 통입니다. 다시 시도하세요.` 문구 출력.
* **결과 저장**: `current_trash.jpg` 이미지와 `trash_type.txt` 텍스트 파일을 로컬에 저장하는 옵션 제공.

---

## 사용법

1. **필수 패키지 설치**

   ```bash
   pip install ultralytics mediapipe opencv-python pillow torch
   ```
2. **모델 학습**

   ```python
   from ultralytics import YOLO

   # 사전 학습된 yolov8n-cls 불러오기
   model = YOLO("yolov8n-cls.pt")

   # 커스텀 데이터로 50 에포크 학습
   model.train(
       data="trash_dataset/train",
       epochs=50,
       imgsz=256,
       batch=32,
       lr0=0.01
   )
   ```
3. **실행**

   ```bash
   python trash_classifier.py
   ```

---

## 코드 구성 및 파일 설명

```
RecyclingTrashClassifier/
├── runs/
│   └── best.pt                # 학습된 YOLOv8 모델
├── images/
│   ├── 플라스틱.png
│   ├── 종이.png
│   └── 캔.png
├── trash_dataset/
│   ├── train/
│   └── val/                   # 이미지 분류용 데이터
├── trash_classifier.py        # 통합 실행 파일
├── train_trash.py             # 학습 파일
├── current_trash.jpg          # 분류 결과 이미지
├── trash_type.txt             # 분류 결과 클래스명 저장
└── README.md
```
* **trash_classifier.py**

  * 실시간 쓰레기 분류 (YOLOv8 + Voting)
  * 손 제스처 기반 드래그
  * GUI 기반 인터페이스 (Tkinter)
  * 분류 결과 이미지/텍스트 저장 및 안내문 출력 포함
* **train_trash.py**
  
  * Ultralytics YOLOv8 분류 모델 학습용 코드
  * 커스텀 데이터셋(trash_dataset/)을 사용하여 fine-tuning
  

---

## 성능 평가

실제 데이터셋에서 학습된 모델의 성능을 다음과 같이 평가하였습니다:

1. **Confusion Matrix**

<img src="https://github.com/user-attachments/assets/5db10902-d9db-40fc-b915-f3bb1bb36250" width="50%">

| 예측\실제 | can | paper | plastic |
|:---------:|:-----:|:-----:|:-------:|
| **can**   |  98   |   0   |    4    |
| **paper**   |   0   |  93   |   19    |
| **plastic** |   4   |   2   |   60    |

   * 각 클래스(캔, 종이, 플라스틱)에 대한 오차 행렬을 통해 분류 정확도를 정밀 분석
   * 일부 오분류가 존재하지만 전체적으로 높은 정확도
   * plastic 클래스에서 상대적으로 성능이 낮음

   - **can** 클래스는 98% 정확히 예측
   - **paper** 클래스는 93% 정확도를 보임임
   - **plastic** 클래스는 약 91%이 plastic으로 올바르게 예측


2. **Confusion Matrix (Normalized)**
<img src="https://github.com/user-attachments/assets/840e486b-4ddb-4dbd-bb8b-e0aed6b1e586" width="50%">

| 예측\실제 | can | paper | plastic |
|:---------:|:-----:|:-----:|:-------:|
| **can**   | 0.96  | 0.00  | 0.05    |
| **paper**   | 0.00  | 0.98  | 0.23    |
| **plastic** | 0.04  | 0.02  | 0.72    |


   * 클래스별 데이터 수 불균형 영향을 배제하기 위해 정규화된 혼동 행렬을 사용
   * paper → plastic, can → plastic 특정 클래스 간 혼동이 존재

3. **학습 곡선(Training Curves)**
<img src="https://github.com/user-attachments/assets/35a5658b-a4a7-43fc-8322-00978fc8160a" width="50%">

   * Train Loss는 지속적으로 감소하여 안정적인 학습 확인
   * validation Loss는 후반부에 overfitting 경향 보임
   * Top-1 정확도는 약 89%에서 수렴


---

## 한계점

* **환경 민감성**: 조명, 배경 복잡도에 따라 MediaPipe 및 분류 모델 성능이 달라짐
* **실시간 처리 속도**: CPU 에서 대용량 모델 사용 시 프레임 드랍 발생 가능
* **클래스 확장**: 새로운 쓰레기 종류 추가 시 데이터 수집, 모델 재학습 필요
* **제스처 인식 오류**: 손 가려짐, 빠른 손 움직임에 취약
---

## 향후 계획

* **추론 최적화**: TensorRT, ONNX Runtime 적용으로 응답 시간 단축
* **멀티플랫폼 배포**: 웹 앱, 모바일 앱으로 확장
* **사용자 경험 개선**: 음성 안내, 터치스크린 제스처 추가, 통계 대시보드 제공
* **데이터 자동 보강**: 사용자 환경에서 수집된 샘플 수집 및 준지도 학습 파이프라인 구축
