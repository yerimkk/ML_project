# 니켈 시장 분석 및 공급위기 예측

## 📌 프로젝트 개요


### 배경
1. **산업적 중요성**: 니켈은 첨단 산업에 필수적인 핵심 광물
2. **시장 변동성**: 니켈 시장은 공급망의 변화와 정책적 요인에 의해 큰 변동성을 보임
3. **경제적 영향**: 니켈의 가격 변동과 금융의 불안정성은 기업의 수익성과 국가경제에 직접적인 영향을 미침
4. **미래 예측 필요성**: 한국은 니켈 수입의존도가 높기 때문에, 안정적인 니켈 수급을 위해 가격과 공급망을 조기에 예측하고 대비하는 것이 중요


### 목표
니켈 시장의 주요 변동 요인을 분석하고, 주요 변수(니켈 가격, 생산량, 소비량, 재고량, 환율)를 바탕으로 공급 위기 여부를 예측


### 진행 과정
1. **데이터 수집 및 전처리**
2. **EDA**: 
    - 데이터 상관관계 분석 및 시각화
    - 글로벌 니켈 공급망 및 관련 산업 동향 분석
    - 니켈 가격 변동 원인 분석
3. 강성분석 : 뉴스기사 감정분석
   - 특정 기간의 긍정적/부정적인 기사 비율 확인
   - 니켈 시장 분위기의 변화 추이 분석
4. 시계열 예측 모델링 : Prophet & SARIMA
   - 니켈 가격 예측 및 과거 트렌드 분석
   - 재고량, 활용, 소비량, 생산량 예측
   - Prophet, SARIMA 모델 비교 ⇒ SARIMA 모델 선정
   
5. 이진분류 모델링 : 공급위기 탐지
   - 수급안정화지수를 기준으로 이진분류 
   - Random Forest, XGBoost 모델 사용하여 학습 
   - 하이퍼파라미터 튜닝 후 교차검증
   - 모델 비교 ⇒ Random Forest 모델 선정
   - 2024년 니켈 공급위기 예측
     
### 데이터 수집

공공데이터 포털
- 광종별 국가별 수출입 현황
- 국가별 니켈 소비량
- 국가별 니켈 생산량

KOMIS 한국자원정보서비스
- 니켈 가격 데이터
- 니켈 수급안정화지수
- LME(런던금속거래소) 니켈 재고량
- KOMIS 일일 자원 뉴스

Yahoo Finance
- 원/달러 환율

### EDA
니켈 가격 변동 주요 요인
- 인도네시아의 생산량 증가
- 코로나 19, 전쟁과 같은 지정학적 리스크로 인한 시장 불확실성 
- 전기차 수요 변화

⇒ 결론 : 
니켈 가격은 주요 생산국의 정치적, 경제적 상황과 전기차 시장의 변화에 크게 영향을 받음
이러한 변동성에 영향을 미치는 시장 심리를 더 잘 이해하기 위해, 니켈 관련 뉴스 기사 감정 분석을 수행

### 뉴스데이터 감성분석
금융데이터에서 높은 성능을 보이는 ProsusAI의 FinBERT 모델을 사용하여 니켈 시장 분위기의 변화 추이 분석
![image](https://github.com/user-attachments/assets/3b0cee4f-6069-4f65-8857-b991bf98bb83)

2020년 상반기, 2022년 상반기에 부정적 감정이 우세
• 2020년 상반기 : 팬데믹으로 인한 시장의 우려를 반영
• 2022년 상반기 : 러시아-우크라이나 전쟁으로 인한 니켈 공급 불확실성 반영
⇒ 시장의 심리적 상태가 중요한 글로벌 사건에 따라 크게 변화하며, 이 변화가 니켈 가격 변동에 영향을 미침

### 시계열 예측 모델링 (Prophet & SARIMA)
![image](https://github.com/user-attachments/assets/6579ad4a-b844-4e95-95d3-58ae881c7e7f)

⇒ 전반적으로 좋은 성능을 보인 SARIMA 모델 최종 선정

예측결과
![image](https://github.com/user-attachments/assets/cd99d106-044b-47ba-aa89-4c414547a5d8)

니켈 가격 하락, 생산량·소비량·재고량 증가, 환율 변동성 예상


## 이진분류 모델링 : 공급위기 예측

1. 종합적인 관점에서 공급 위기를 평가하기 위해 여러 요인을 고려한 수급안정화지수를 기준으로 공급 위기 여부를 분류
- 20 이하 : 공급위기 (1)
- 21-100: 공급급안정 (0)

2. 모델 학습 : Random Forest, XGBoost, Logistic Regression
- Feature : 가격, 재고량, 환율, 생산량, 소비량
- Target : 공급위기

3. 하이퍼파라미터튜닝
GridSearchCV로 최적 파라미터를 탐색 후 적용

4. 모델비교
교차검증 평균 정확도 비교
- Random Forest: 0.9404
- XGBoost: 0.9287

5. 모델선정 : Random Forest
- 모든 폴드에서 균일한 성능과 높은 정확도를 보임
- Recall(1 클래스)이 대부분의 폴드에서 높은 값을 유지 

6. 예측 결과
SARIMA로 예측한 피처 값을 기반으로 공급위기 예측 
⇒ 니켈 수급안정


### 📌 2024 상반기 실제 데이터와 예측 결과 비교 
2024년 실제 데이터가 존재하는 니켈 가격, 재고량, 환율, 수급 안정화지수를 비교하여 모델의 예측 성능 및 공급 위기 예측의 신뢰성 평가

![image](https://github.com/user-attachments/assets/22c8a129-722e-46f8-aa1e-78f325f01326)



① 니켈 가격

실제 가격은 5월까지 급상승 : 연이은 니켈 조업 중단으로 인한 공급 차질 때문인 것으로 보임

② 재고량  

실제 재고량은 예상치를 훨씬 벗어나 9만 까지 상승

③ 원/달러 환율 

1,270원에서 1,310원 사이에서 변동할 것으로 예측했으나 실제 환율은 지속적인 상승세를 보이며 1,380원까지 상승

④ 수급안정화지수

실제 2024년 상반기 수급안정화지수가 20-60사이로 공급안정

2024년 상반기 예측 성능 평가

![image](https://github.com/user-attachments/assets/ce827de5-0226-4c33-9c2e-ddffb2bc9b9c)


- 종합평가 : 환율과 니켈 가격의 변동 패턴을 제대로 반영하지 못했고, 재고량 예측도 실제 값과 큰 차이를 보임

- 한계점

단변량 SARIMA 모델 : 외부 요인 반영 불가로 복잡한 변동성을 잡지 못함
재고량 예측 : 실제 값과의 큰 차이로 인해 개선 필요



## 📌 개선점

- 정치적 상황, 환경 규제 등 외부 요인 반영

- 금리, GDP 성장률과 같은 거시경제 지표와 니켈 관련 산업별 지표를 추가로 고려

- LSTM 모델 시도 : 복잡한 시계열 패턴을 더 잘 학습하여 예측 성능 개선



