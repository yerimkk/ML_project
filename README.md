# 니켈 시장 분석 및 공급위기 예측

## 📌 프로젝트 개요

### 기간
2024.06.01 ~ 2024.06.28

### 배경
1. **산업적 중요성**: 니켈은 첨단 산업에 필수적인 핵심 광물
2. **시장 변동성**: 니켈 시장은 공급망의 변화와 정책적 요인에 의해 큰 변동성을 보임
3. **경제적 영향**: 니켈의 가격 변동과 금융의 불안정성은 기업의 수익성과 국가경제에 직접적인 영향을 미침
4. **미래 예측 필요성**: 한국은 니켈 수입의존도가 높기 때문에, 안정적인 니켈 수급을 위해 가격과 공급망을 조기에 예측하고 대비하는 것이 중요


![image](https://github.com/user-attachments/assets/af72ca2f-a1cf-4354-abc1-81097bb97889)



### 목표
니켈 시장 동향 분석 및 예측 데이터를 활용한 공급 위기 탐지

### 진행 과정
1. **데이터 수집 및 전처리**
2. **EDA**: 
    - 데이터 상관관계 분석 및 시각화
    - 글로벌 니켈 공급망 및 관련 산업 동향 분석
    - 니켈 가격 변동 원인 분석
3. **머신러닝 (1)**: NLP 뉴스기사 감성분석
   - 특정 기간의 긍정적/부정적인 기사 비율 확인
   - 니켈 시장 분위기의 변화 추이 분석
4. **머신러닝 (2)**: Prophet & ARIMA
   - 니켈 가격 예측 및 과거 트렌드 분석
   - 재고량, 활용, 소비량, 생산량 예측
   - Prophet, ARIMA 모델 비교 ⇒ Prophet 모델 선정
   
5. **머신러닝 (3)**: 이진분류 모델을 활용한 공급위기 탐지
   - 수급안정화지수를 기준으로 이진분류 
   - Random Forest, XGBoost, Logistic Regression 모델 사용하여 학습 
   - 하이퍼파라미터 튜닝 후 교차검증
   - 모델 비교 ⇒ Logistic Regression 모델 선정
   - 2024년 니켈 공급위기 예측

## 📌 분석 결과

### 2024년 예측: 니켈 가격 하락 + 수급안정

![image](https://github.com/user-attachments/assets/0764a221-14da-4351-a74e-5e82b3158a27)


- **2024년 니켈 가격 그래프**
  상반기에 니켈 가격이 하락할 것으로 예상했으나 실제 니켈 가격은 연초부터 상승하다가 5월 말에 하락. 하반기에는 과잉 공급과 수요 둔화로 인해 연말에 하락할 것으로 예상됨.

- **2024 상반기 공급안정화지수 그래프**
  상반기에도 안정적인 수준을 유지할 것으로 보임. 하반기에도 안정적인 수준을 유지할 것으로 보임.

## 📌 개선점

주요 생산국에 대한조사와 함께 금리, GDP 성장률, 인플레이션율과 같은 거시경제지표나 니켈관련
산업별 지표를 추가적으로 고려한 좀 더 정교한 분석이 필요해 보임

