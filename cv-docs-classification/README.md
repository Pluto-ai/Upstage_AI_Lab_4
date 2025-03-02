<img width="653" alt="image" src="https://github.com/user-attachments/assets/885f5111-8288-4b6d-a43a-797cac8a9465">

# 프로젝트 이름
- 문서 분류 태스크
<br>

## 프로젝트 소개
- 이 프로젝트는 문서 분류 모델을 개발하는 것입니다.
- 이 문서 분류 모델은 품질 저하된 17종의 문서를 분류할 수 있도록 설계됩니다. 
<br>

## 팀원 구성

<div align="center">

| **팀장** | **팀원 1** | **팀원 2** | **팀원 3** | **팀원 4** |
| :------: |  :------: | :------: | :------: | :------: |
|[<img src="https://avatars.githubusercontent.com/u/156163982?v=4" height=150 width=150> <br/> @Github]([https://github.com/misule0423]) |[<img src="https://avatars.githubusercontent.com/u/50396041?v=4" height=150 width=150> <br/> @Github](https://github.com/soomin8442) |[<img src="https://avatars.githubusercontent.com/u/156163982?v=4" height=150 width=150> <br/> @Github](https://github.com/) |[<img src="https://avatars.githubusercontent.com/u/156163982?v=4" height=150 width=150> <br/> @Pluto-ai](https://github.com/pluto-ai) |[<img src="https://avatars.githubusercontent.com/u/1223020?v=4" height=150 width=150> <br/> @deptno](https://github.com/deptno) |
</div>

<br>

## 1. 개발 환경

- 주 언어 : Python
- 주 프레임워크 : Pytorch
- 협업 툴 : Github, Slack, Zoom
<br>

## 2. 채택한 개발 기술

### Pytorch, torchvision, scikit-learn

- Pytorch 프레임워크
  - 딥러닝 파이썬 프레임워크
    
- torchvision, pillow
  - 컴퓨터비전용 데이터셋, 모델, 이미지 전처리 제공
  - 이미지 라이브러리
    
- scikit-learn
  - 성능 평가 지표
    
### effdet, efficientnet_pytorch

- effdet
  - efficientDet 객체 탐지 모델 라이브러리
    
- efficientnet_pytorch_b4
  - efficientNet 분류 모델 라이브러리

- ConvNext_tiny
  - ConvNex_tiny 분류 모델 라이브러리


<br>

## 3. 프로젝트 구조
```
├── README.md
├── .gitignore
├── data_augmentation.ipynb
└── ensemble.ipynb

```

<br>

## 4. 역할 분담

### 이민석
- **역할**
    - 조장, 데이터 전처리, 모델 구축 및 학습, 앙상블
- **기능**
    - 데이터 증강, 분류 모델 구축 및 학습, 앙상블 개발
<br>

### 문수민
- **역할**
    - 데이터 전처리, 모델 구축 및 학습, 실험관리
- **기능**
    - 데이터 증강, 분류 모델 구축 및 학습, 실험관리
<br>

### 김민수
- **역할**
    - 프로젝트를 진행하며 맡은 역할 작성
- **기능**
    - 프로젝트를 진행하며 개발한 기능 작성
<br>

### 오승민
- **역할**
    - 데이터 전처리, 모델 구축 및 학습
- **기능**
    - 데이터 증강, 분류 모델 구축 및 실습, 모델 파라미터 조정
<br>

### 이봉균
- **역할**
    - 데이터 전처리, 모델 구축 및 학습, 앙상블
- **기능**
    - 데이터 증강, 분류 모델 구축 및 학습
<br>

## 5. 개발 기간 및 작업 관리

### 개발 기간
- 전체 개발 기간 : 2024-10-28 ~ 2024-11-09
- 기능 구현 : 2024-11-04 ~ 2024-11-09

## 6. 프로젝트 개요
### EDA
![image](https://github.com/user-attachments/assets/debde63f-86de-4220-b46a-ff6d5c3cc123)

![image](https://github.com/user-attachments/assets/96053b27-fa0f-4f71-87cf-2653b039a1db)

![image](https://github.com/user-attachments/assets/87e631be-5705-4495-ac07-a3a273cd57c4)

### 데이터 구성
![image](https://github.com/user-attachments/assets/0b7917ce-9080-4b46-b3cb-d8f0291e3040)

### 모델 구성
**전체 모델 구성**
![image](https://github.com/user-attachments/assets/11039b23-e9b9-4b11-bdfd-9e83fd3fc654)

![image](https://github.com/user-attachments/assets/831aad00-4177-48fb-ac06-2fa624bfbdab)

**모델별 구성**
![image](https://github.com/user-attachments/assets/4a8ba656-e13f-4a66-b3db-6b51af07e45d)

**EfficientNet_b4모델, 손실함수 및 3/4클래스에 대한 online augmentation**
  ![image](https://github.com/user-attachments/assets/6a0075d6-fb9b-4130-864f-ec8ce947fe6d)
  
### 최종 결과
![image](https://github.com/user-attachments/assets/980c77cc-f194-4290-98eb-0dff86fb5d43)

## 7. 프로젝트 인사이트
### OCR을 활용한 클래스 분류
![image](https://github.com/user-attachments/assets/a87a3a42-dbf5-4f2c-91ae-44a321081409)

### Finetuning을 통한 성능향상 시도
![image](https://github.com/user-attachments/assets/46381225-c092-4d20-a4a0-366e7e647713)


<br>

## 8. 프로젝트 회고

### 이민석
문서 분류 태스크에서 데이터 분석, 데이터 증강, 모델 구조 선택, 학습, 앙상블 전체를 다루어 볼 수 있어서 좋았습니다.

### 문수민
CV 관련 대회는 이번이 처음이었으며, 이번 프로젝트를 통해 다양한 이미지 증강과 처리 방법을 배울 수 있었습니다. 모델 부분에서는 사전 학습된 모델을 활용해 직접 파인튜닝을 진행하며 성능 변화를 확인했습니다. 또한, 여러 모델을 사용하고 앙상블을 적용해 성능 개선을 이루었습니다. 실험 관리 측면에서는 가설을 세우고 결과를 확인하면서, 반복적인 개선을 통해 더 나은 방법을 찾아가는 경험을 할 수 있었습니다.

### 김민수
CV 프로젝트를진행하면서많이배웠습니다
<br>

### 오승민
CV 관련 모델들이 정말 다양하게 있고, 어느 특정 모델만 사용하는 것이 아니라 진행할 프로젝트 주제에 따라 다르게 선택할 수 있다는 점이 흥미로웠습니다
<br>

### 이봉균
문서 분류 태스크를 통해 데이터 증강부터 모델 학습, 앙상블까지 전체를 경험할 수 있어 좋았습니다.
<br>

