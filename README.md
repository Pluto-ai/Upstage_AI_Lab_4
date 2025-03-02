# Title (Please modify the title)
## Team

| ![김요셉](https://avatars.githubusercontent.com/u/175805693?v=4) | ![이주하](https://avatars.githubusercontent.com/u/45289805?v=4) | ![변해영](https://avatars.githubusercontent.com/u/165775145?v=4) | ![오승민](https://avatars.githubusercontent.com/u/177705512?v=4) | ![김동규](https://avatars.githubusercontent.com/u/102230809?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김요셉](https://github.com/sebi0334)             |            [이주하](https://github.com/jl3725)             |            [변해영](https://github.com/jenny20240401)             |            [오승민](https://github.com/Pluto-ai)             |            [김동규](https://github.com/Lumiere001)             |
|                            팀장, 데이터 증강                              |                            backtranslation 및 gemma, prompt engineering                              |                            데이터 증강                             |                            model freezing                             |                            T5 실험 및 optuna 활용                             |

## 0. Overview
### Environment
- Python, Pytorch

### Requirements
- requirements.txt 참고

## 1. Competiton Info

### Overview

- _일상 대화 요약_

일상 대화는 회의, 토의, 사소한 대화를 포함해 다양한 주제와 관점을 주고받는 과정입니다. 
하지만 대화를 녹음하더라도 전체를 다시 듣는 것은 비효율적이기에 요약이 필요합니다. 
대화 중 요약은 집중을 방해하고, 이후 기억에 의존한 요약은 오해와 누락을 초래할 수 있습니다. 
이를 해결하고자, 이번 대회에서는 일상 대화를 기반으로 자동 요약문을 생성하는 모델을 개발합니다.


### Timeline

- ex) November 18, 2024 - Start Date
- ex) November 28, 2024 - Final submission deadline

## 2. Components

### Directory

```
├── README.md
├── docs
├── baseline_ES.ipynb
├── T5-base.py
├── baseline.ipynb
├── requirements.txt
|── 4기 NLP_경진대회_1조.pdf
```

## 3. Data descrption

### Dataset overview

최소2턴, 최대 60턴으로 대화가 구성되어 있습니다. 대화(*dialogue)를 보고 이에 대한 요약(*summary) 를 예측하는 것이 최종 목표입니다.

- fname : 대화 고유번호 입니다. 중복되는 번호가 없습니다.

- dialogue : 최소 2명에서 최대 7명이 등장하여 나누는 대화 내용입니다. 각각의 발화자를 구분하기 위해#Person”N”#: 을 사용하며, 발화자의 대화가 끝나면 \n 으로 구분합니다. 이 구분자를 기준으로 하여 대화에 몇 명의 사람이 등장하는지 확인해보는 부분은 EDA 에서 다루고 있습니다.

- summary : 해당 대화를 바탕으로 작성된 요약문입니다.

### EDA

- 대화별 주제로 나눠서 데이터 증강함. baseline.ipunb 파일에 잘 나와 있음.

### Data Processing

- Solar pro API를 활용해서 데이터 증강함.

## 4. Modeling

### Model descrition

- BART(Bidirectional and Auto-Regressive Transformers)는 Facebook AI(현재는 Meta AI)에서 2019년에 개발한 자연어 처리(NLP) 모델.
- BART는 인코더-디코더(Encoder-Decoder) 아키텍처를 기반, BERT와 GPT의 장점을 결합한 모델로 설계. 
- 텍스트 생성, 이해에서 좋은 성능 발휘.

## 5. Result

### Leader Board

- 전체 6팀 중 6등

### Presentation

- 첨부된 발표 자료 참고

## etc

### Meeting Log

- 대회 마지막주 매일 오후 2시 모여서 미팅 진행

### Reference

- [KETI-AIR 모델 참고 주소](https://huggingface.co/KETI-AIR/ke-t5-base)

## 느낀점

### 김요셉

베이스라인 코드가 다른 대회보다 점수가 높고 NLP 대회여서 그런지 이전 대회보다 점수를 향상시키기 휠씬 어려웠습니다. 파라미터를 조금만 수정해도 메모리가 부족한 문제는 다양한 실험을 하는데 발목을 잡았습니다. 팀 내 에서 다양한 방법을 수행해보았지만 유의미한 결과가 없어 방향성에 대해 많이 고민하는 대회였습니다. 

### 이주하

여태 모든 대회 및 프로젝트에서 가장 메모리 문제가 가장 심했던 거 같음. 그리고 워낙 베이스라인 자체가 성능이 잘 나오다 보니 웬만한 실험을 해도 올리기 힘들었음. 영어였으면 훨씬 나았을텐데 한국말이 아 다르고 어 다르단 걸 다시 한번 느꼈음.

### 변해영

NLP는 어렵네요..데이터 증강에도 시간이 오래 걸리고..

### 오승민

생각했던 성능 개선 방안들이 이번 프로젝트에서 맞지 않아서 그런건지 생각보다 성능 개선이 이루어 지지 않아서 많이 아쉬웠습니다.

### 김동규

이번 대회는 상당히 어려웠습니다. 메모리 부족 문제로 원하는 모델로 실험하기도 어려웠고, 각 코드가 컴퓨터의 메모리에 미치는 영향도 바르게 알지 못했고, 체계적인 가설 세우기와 실험관리를 할 수 있는 능력이 없다보니 모든 가설과 실험마다 실패를 반복했습니다. 그리고 그 어떤 대회보다 실험 실패시 시간 비용이 많이 들어갔습니다.

이번 대회를 통해 부족한 부분을 발견할 수 있어서 좋았고, 앞으로 발전할 날들이 기대가 됩니다:)