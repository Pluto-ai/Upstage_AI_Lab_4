# Title (Please modify the title)
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이봉균](https://avatars.githubusercontent.com/u/1223020?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: |:-----------------------------------------------------------:| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |           [이패캠](https://github.com/UpstageAILab)            |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                         git                          |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Requirements
matplotlib==3.7.1
numpy==1.23.5
pandas==1.5.3
scipy==1.11.3
seaborn==0.12.2
scikit-learn==1.2.2
statsmodels==0.14.0
tqdm==4.66.1
eli5==0.13.0

## 1. Competiton Info

### Overview
House Price Prediction 경진대회는 주어진 데이터를 활용하여 서울의 아파트 실거래가를 효과적으로 예측하는 모델을 개발하는 대회입니다. 

부동산은 의식주에서의 주로 중요한 요소 중 하나입니다. 이러한 부동산은 아파트 자체의 가치도 중요하고, 주변 요소 (강, 공원, 백화점 등)에 의해서도 영향을 받아 시간에 따라 가격이 많이 변동합니다. 개인에 입장에서는 더 싼 가격에 좋은 집을 찾고 싶고, 판매자의 입장에서는 적절한 가격에 집을 판매하기를 원합니다. 부동산 실거래가의 예측은 이러한 시세를 예측하여 적정한 가격에 구매와 판매를 도와주게 합니다. 그리고, 정부의 입장에서는 비정상적으로 시세가 이상한 부분을 체크하여 이상 신호를 파악하거나, 업거래 다운거래 등 부정한 거래를 하는 사람들을 잡아낼 수도 있습니다. 

저희는 이러한 목적 하에서 다양한 부동산 관련 의사결정을 돕고자 하는 부동산 실거래가를 예측하는 모델을 개발하는 것입니다. 특히, 가장 중요한 서울시로 한정해서 서울시의 아파트 가격을 예측하려고합니다.

참가자들은 대회에서 제공된 데이터셋을 기반으로 모델을 학습하고, 서울시 각 지역의 아파트 매매 실거래가를 예측하는데 중점을 둡니다. 이를 위해 선형 회귀, 결정 트리, 랜덤 포레스트, 혹은 딥 러닝과 같은 다양한 regression 알고리즘을 사용할 수 있습니다.


제공되는 데이터셋은 총 네가지입니다. 첫번째는 국토교통부에서 제공하는 아파트 실거래가 데이터로 아파트의 위치, 크기, 건축 연도, 주변 시설 및 교통 편의성과 같은 다양한 특징들을 포함하고 있습니다. 두번째와 세번째 데이터는 추가 데이터로, 서울시에서 제공하는 지하철역과 버스정류장에 대한 다양한 정보들을 포함하고 있습니다. 마지막 네번째 데이터는 평가 데이터로, 최종 모델성능에 대한 검증을 위해 사용됩니다.

참가자들은 이러한 다양한 변수와 데이터를 고려하여 모델을 훈련하고, 아파트의 실거래가에 대한 예측 성능을 높이기 위한 최적의 방법을 찾아야 합니다.

경진대회의 목표는 정확하고 일반화된 모델을 개발하여 아파트 시장의 동향을 미리 예측하는 것입니다. 이를 통해 부동산 관련 의사 결정을 돕고, 효율적인 거래를 촉진할 수 있습니다. 또한, 참가자들은 모델의 성능을 평가하고 다양한 특성 간의 상관 관계를 심층적으로 이해함으로써 데이터 과학과 머신 러닝 분야에서의 실전 경험을 쌓을 수 있습니다.


### Timeline

- 2024.09.02 10:00
~
2024.09.13 13:00

## 2. Components

### Directory

- _Insert your directory structure_
```sh 
.
├── Pipfile
├── Pipfile.lock
├── README.md
├── code                            # ipynb
│   ├── XGBoost.ipynb
│   ├── baseline_code.ipynb
│   ├── baseline_code_LGBM.ipynb
│   ├── deptno                            
│   │   ├── load_log_model.py       # 저장된 pkl 로 모델을 불러와 test.csv 로 부터 예측
│   │   ├── run.py                  # optuna 를 통해 LGBRegressor 하이퍼 파라메터를 찾고 가장 좋은 모델, 파라메터를 기록
│   │   ├── run_with_log.py         # 모델 학습시 target 에  log scale 을 적용한다
│   │   ├── train-vs-test-xy.ipynb
│   │   └── train-vs-test.ipynb
│   ├── deptno.ipynb
│   └── requirements.txt
└── data                            # 원본 데이터, 및 전처리를 거진 데이터
    ├── X_test_6.csv
    ├── X_train.5.2.csv
    ├── bus_feature.csv
    ├── sample_submission.csv
    ├── selected_xy.csv
    ├── subway_feature.csv
    ├── test.csv
    ├── test_xy.csv
    ├── train.csv
    ├── train_xy.csv
    └── y_train.5.2.csv
```

## 3. Data descrption

### Dataset overview

input : 9,272개의 아파트 특징 및 거래정보

output : 9,272개의 input에 대한 예상 아파트 거래금액

### Data Processing

[기본 제공 데이터]
train, test
결측치가 많은 좌표데이터에 대해서 수작업(도로명주소 누락, 오기입 수정)과 오픈API(국토부 VWorld) 사용을 병행해 전부 채움
이후 기본적인 처리 후 사용

subway_feature
1.5km 이내의 가장 가까운 지하철역 이름, 노선명,
1.5km 이내에 존재하는 모든 지하철 역의 최단, 최장, 평균 거리
노선별 우선순위를 정의해 여러 개의 지하철역이나 노선이 선택될 경우, 우선순위가 높은 노선을 선택


[추가로 사용한 데이터]
school_utf8(공공데이터포털) : 서울 학교 데이터
1km 이내의 가장 가까운 초등학교, 중학교, 고등학교 까지의 거리
가장 가까운 등급별(초,중,고) 학교명

modified_hotel_data(서울정보소통광장) : 서울 숙박업 데이터 (5성, 4성 호텔 데이터만 사용)
1km 이내에 존재하는 5성급, 4성급 호텔의 수

modified_store_data(서울 열린데이터 광장) : 서울 대규모점포 인허가 데이터 (백화점 데이터만 사용)
5km 이내에 존재하는 가장 가까운 백화점까지의 거리
5km 이내에 존재하는 모든 백화점의 수

gdp_data(서울 열린데이터 광장) : 서울 구별 GDP 데이터
부동산 계약년도의 자치구별 지역내총생산, 인구, 1인당 지역내총생산, 소득수준지수(서울특별시=100)

## 4. Modeling

### Model descrition

XGBoost 선택 이유
부동산 가격은 여러가지 요인의 영향을 받아 비선형적인 모습을 보인다고 생각했고,
데이터를 살펴봤을 때 이상치가 어느정도 있다고 판단해
비선형성과 이상치 처리에 강점을 보이는 XGBoost를 선택함

XGBoost 파라미터
목적함수로 MSE를 최소화하는 목적함수인 reg:squarederror 사용
-> RMSE가 MSE의 제곱근이므로 MSE를 최소화하는 것이 결과적으로 RMSE를 최소화하는 효과가 있음
트리의 최대 깊이와 subsample, colsample_bytree를 조절해 과적합 방지 
학습률을 0.01로 설정해 모델이 좀 더 안정적으로 학습하도록 함
학습률이 낮은 것을 감안해 최대 반복 횟수를 10000으로 설정했고,
혹시라도 성능 개선이 되지 않는데도 지속해서 학습하는 것을 방지하기 위해
early_stopping_rounds를 50 으로 설정함

### Modeling Process

[유의미해 보이는 feature importances]
강남여부
1인당 GDP (gdp_per_capita_thousand_krw) -> 단위: 천원
전용면적, 평
구
좌표X, Y 같은 위치 데이터
5km 이내의 백화점 수 (department_count_5km)
계약년도
건축년도
가장 가까운 백화점까지의 거리 (closest_department_dist)
지역내총생산 (gdp_million_krw) -> 단위: 백만원

[best 100, worst 100 분포 비교]
전용면적이 넓은 쪽의 가격을 예측하는 성능이 좋지 않음을 확인할 수 있음
-> 이 문제를 해결하기 위해 전용면적이 좁은 데이터에는 Light GBM을,
전용면적이 넓은 데이터에는 XGBoost를 적용하는 앙상블 기법을 사용하고자 했으나 성능이 더 좋지 않았고,
조정하기엔 시간이 부족해 결과적으로 적용하지 못함

## 5. Result

### Leader Board


