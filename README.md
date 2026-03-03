# ADMT-RPA: AD-MT with Revised Progressive Augmentation

AD-MT (Alternate Dual-Model Training) 프레임워크에 개선된 점진적 증강 전략을 적용한 반지도 의료 영상 분할 구현입니다.

## 개요

본 연구는 원본 [AD-MT 논문](https://github.com/zhenzhao/AD-MT)을 확장하여 두 Teacher 모델 간 증강 전략을 동기화하는 **Revised Progressive Augmentation (RPA)** 방식을 도입했습니다.

### 핵심 수정사항

#### 기존 AD-MT 방식
- **Teacher Model A2**: Copy-Paste와 CutMix 증강 모두 적용
  - Copy-Paste → Stage 1
  - CutMix → Stage 2
- **Teacher Model A1**: Color Jitter만 적용

#### ADMT-RPA 방식
- **Teacher Model A2**: Dice Score 정체 기반으로 Copy-Paste → CutMix 점진적 적용
- **Teacher Model A1**: **동기화된 Copy-Paste** + Color Jitter
  - A2가 Copy-Paste 단계(Stage 1)일 때, A1도 Copy-Paste 적용
  - Copy-Paste를 Color Jitter **이전에** 적용하여 용종과 배경 간 극단적인 색상 차이 방지
  - A2가 CutMix로 전환(Stage 2)하면, A1은 Color Jitter만 적용

### 연구 동기

동기화된 Copy-Paste 전략은 용종 분할의 핵심 과제를 해결합니다:
- **문제**: 용종과 배경 간의 심한 색상 불일치가 모델을 오도할 수 있음
- **해결책**: Color Jitter 이전에 Copy-Paste 증강을 적용하여 더 현실적인 학습 샘플 생성
- **이점**: 초기 학습 단계에서 두 Teacher 모델 간 일관된 데이터 증강으로 모델 강건성 향상

## 리포지토리 구조

```
ADMT-RPA/
├── code/
│   ├── A1TCP.py                    # RPA 전략이 적용된 메인 학습 스크립트
│   ├── newtest.py                  # 버그 수정이 반영된 테스트 스크립트
│   ├── train_utils.py              # 학습 유틸리티
│   ├── val_2D.py                   # 2D 검증 함수
│   ├── dataloaders/
│   │   ├── mixaugs.py             # 수정됨: Copy-Paste 및 CutMix 구현
│   │   └── dataset_2d.py          # 수정됨: 동기화된 증강이 적용된 데이터셋
│   ├── networks/
│   │   └── net_factory.py         # 모델 생성 팩토리
│   └── utils/
│       ├── losses.py              # 손실 함수
│       ├── ramps.py               # 학습률 Ramp 함수
│       └── util.py                # 일반 유틸리티
├── cfgs/
│   └── config_2d_kvasir_5percent.yml  # Kvasir 데이터셋 설정 파일
├── train.py                        # 수정됨: 학습 런처 스크립트
└── README.md                       # 본 문서
```

## 시작하기

### 요구사항

- Python 3.8+
- PyTorch 1.10+
- CUDA (권장)

### 설치

```bash
# 리포지토리 클론
git clone https://github.com/minsk0429/ADMT-RPA.git
cd ADMT-RPA

# 의존성 설치
pip install -r requirements.txt
```

### 데이터셋 준비

원본 AD-MT 리포지토리 구조를 따라 Kvasir-SEG 데이터셋을 준비하세요:
```
data/
└── Kvasir/
    ├── train/
    ├── test/
    ├── train.list
    └── test.list
```

### 학습

```bash
# 학습 런처 스크립트 사용
python train.py --base_path ./repo/AD-MT \
                --data_path ./data/Kvasir \
                --python_exe python \
                --poly

# 또는 학습 스크립트 직접 실행
python code/A1TCP.py --gpu_id 0 \
                     --cfg cfgs/config_2d_kvasir_5percent.yml \
                     --root_path ./data/Kvasir \
                     --labeled_num 44 \
                     --exp Kvasir/A1TCP_44_v2 \
                     --model unet \
                     --max_iterations 30000
```

### 테스트

```bash
python code/newtest.py --root_path ./data/Kvasir \
                       --model_path ./model/best_model.pth \
                       --model unet \
                       --num_classes 2 \
                       --gpu_id 0
```

## 주요 구성요소

### 1. Adaptive Augmentation Scheduler (A1TCP.py)
- Validation Dice Score의 정체(plateau) 감지를 모니터링
- 개선이 정체될 때 Copy-Paste에서 CutMix로 자동 전환
- A1과 A2 증강 전략 동기화

### 2. Synchronized Copy-Paste (dataloaders/mixaugs.py)
- 크기 제약(이미지의 5%-45%)이 있는 색상 인식 Copy-Paste 구현
- 붙여넣기 전 용종 색상 유사도 보장
- Color Jitter 이전에 적용되어 현실적인 색상 분포 유지

### 3. 버그 수정 (newtest.py, val_2D.py)
- Validation의 Dice 계산 버그 수정 (이전에는 Ground Truth와 관계없이 빈 예측에 대해 1.0 반환)
- True Negative 및 False Positive/Negative를 올바르게 처리하도록 메트릭 계산 개선

## 수정된 파일

### 핵심 변경사항
1. **A1TCP.py** (신규)
   - RPA 전략을 구현한 메인 학습 스크립트
   - Adaptive Augmentation Scheduler
   - 동기화된 Copy-Paste 적용

2. **train.py** (수정)
   - A1TCP 실험 실행 지원 추가
   - Poly Learning Rate Decay 옵션

3. **newtest.py** (신규)
   - Dice 계산 버그 수정
   - 이전 메트릭과 새 메트릭 비교 모드 추가

### 데이터 증강
4. **dataloaders/mixaugs.py** (수정)
   - 색상 유사도 체크가 추가된 Copy-Paste
   - 크기 제약이 있는 Copy-Paste (5%-45% 비율)

5. **dataloaders/dataset_2d.py** (수정)
   - 동기화된 증강 파이프라인 통합
   - Copy-Paste → Color Jitter 순서 적용

### 검증
6. **val_2D.py** (수정)
   - Dice 계산에서 빈 예측 처리 수정

## 실험 설정 및 결과

### 실험 환경
- **데이터셋**: Kvasir-SEG (용종 분할)
- **라벨 데이터**: 44, 88, 176개 이미지 (~5%, ~10%, ~20% of training set)
- **모델**: U-Net
- **최대 반복**: 30,000
- **배치 크기**: 24 (12 labeled + 12 unlabeled)
- **증강 파라미터**:
  - Copy-Paste 확률: 100%
  - CutMix 확률: 100%
  - Color Jitter: (0.5, 0.5, 0.5, 0.25) with p=0.8
  - Gaussian Blur: σ=(0.1, 2.0) with p=0.2
  - 용종 크기 제약: 이미지 면적의 5%-45%

### 실험 결과

#### 라벨 데이터 44개 (5%)

| 실험 | Dice Score | ASD | Copy-Paste → CutMix 전환 시점 |
|------|------------|-----|---------------------------|
| A1TCP_44_v1 | **0.8164** | 27.46 | Iteration 13,455 (Dice: 0.673) |
| A1TCP_44_v2 | 0.8023 | 29.80 | Iteration 12,765 (Dice: 0.697) |

- **분석**: v1이 v2보다 1.41% 높은 Dice Score 달성
- v1은 더 늦게 CutMix로 전환했으며(약 690회 더 늦음), 이는 Copy-Paste 단계에서 더 충분한 학습이 이루어졌음을 시사

#### 라벨 데이터 88개 (10%)

| 실험 | Dice Score | ASD | Copy-Paste → CutMix 전환 시점 |
|------|------------|-----|---------------------------|
| A1TCP_88_v1 | 0.8216 | 25.07 | Iteration 12,078 (Dice: 0.706) |
| A1TCP_88_v2 | **0.8252** | **23.66** | - (전환 없음) |

- **분석**: v2가 v1보다 0.36% 높은 Dice Score 달성
- v2는 CutMix로 전환하지 않고 Copy-Paste만으로 더 나은 성능 달성
- 88개 라벨 데이터에서는 Copy-Paste가 충분히 효과적임을 시사

#### 라벨 데이터 176개 (20%)

| 실험 | Dice Score | ASD | Copy-Paste → CutMix 전환 시점 |
|------|------------|-----|---------------------------|
| A1TCP_176_v1 | **0.8476** | **20.33** | Iteration 5,858 (Dice: 0.701) |
| A1TCP_176_v2 | 0.8375 | 24.11 | - (전환 없음) |

- **분석**: v1이 v2보다 1.01% 높은 Dice Score 달성
- v1은 초기에 CutMix로 전환(Iteration 5,858)하여 더 나은 성능 달성
- 176개 라벨 데이터에서는 조기 CutMix 전환이 효과적

### 주요 발견사항

1. **라벨 데이터 양에 따른 전략 차이**
   - 적은 라벨 데이터(44개): 늦은 전환이 유리
   - 중간 라벨 데이터(88개): Copy-Paste만으로도 효과적
   - 많은 라벨 데이터(176개): 조기 전환이 유리

2. **성능 향상**
   - 44개 라벨: 최대 Dice 0.8164 (v1)
   - 88개 라벨: 최대 Dice 0.8252 (v2) - 0.88% 향상
   - 176개 라벨: 최대 Dice 0.8476 (v1) - 2.24% 향상

3. **증강 전환 타이밍의 중요성**
   - 적응적 증강 전환 전략이 고정 전략보다 유연하고 효과적
   - 데이터셋 크기에 따라 최적의 전환 시점이 다름

**원본 리포지토리**: [https://github.com/zhenzhao/AD-MT](https://github.com/zhenzhao/AD-MT)

## 라이센스

본 프로젝트는 원본 AD-MT 리포지토리의 라이센스를 따릅니다. 자세한 라이센스 정보는 [원본 리포지토리](https://github.com/zhenzhao/AD-MT)를 참조하세요.


