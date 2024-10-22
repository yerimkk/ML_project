from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

plt.style.use('seaborn-darkgrid')
# 한국어 폰트 설정 (맑은 고딕 사용)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows에서 '맑은 고딕' 폰트 경로
fontprop = fm.FontProperties(fname=font_path)

# 그래프 시각화 시, 폰트 설정
plt.rc('font', family=fontprop.get_name())

matplotlib.rcParams['axes.unicode_minus']=False

# 정상성 검정 함수
def check_stationarity(timeseries):
    """
    시계열 데이터의 정상성을 검정하는 함수입니다.

    Args:
    - timeseries (pd.Series): 정상성 검정을 수행할 시계열 데이터
    시각화와 Dickey-Fuller 테스트 결과를 출력합니다.
    """
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()
    # rolling statistics plot
    plt.figure(figsize=(12, 6))
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Dickey-Fuller Test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    print(dfoutput)

    # 정상성 여부 출력
    if dftest[1] <= 0.05: # p-value가 0.05 미만이면 정상성 확보
        print('데이터는 정상성을 가집니다.')
    else:
        print('데이터는 정상성을 가지지 않습니다.')


# 생산량, 소비량 정상성 검정 함수
def check_stationarity_production_consumption(timeseries):
    """
    생산량 또는 소비량 데이터의 정상성을 검정하는 함수입니다.

    Args:
    - timeseries (pd.Series): 정상성 검정을 수행할 시계열 데이터
    시각화와 Dickey-Fuller 테스트 결과를 출력합니다.
    """
    rolmean = timeseries.rolling(window=3).mean()
    rolstd = timeseries.rolling(window=3).std()

    # rolling statistics plot
    plt.figure(figsize=(12, 6))
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Dickey-Fuller Test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    print(dfoutput)

    # 정상성 여부 출력
    if dftest[1] <= 0.05: # p-value가 0.05 미만이면 정상성 확보
        print('데이터는 정상성을 가집니다.')
    else:
        print('데이터는 정상성을 가지지 않습니다.')       


# 차분 및 Dickey-Fuller 테스트 함수
def differencing_and_adf_test(df):
    """
    시계열 데이터에 대해 1차 차분을 수행하고 Dickey-Fuller 테스트를 통해 정상성을 검정하는 함수입니다.

    Args:
    - df (pd.Series): 차분을 수행할 시계열 데이터

    Returns:
    - pd.Series: 차분된 시계열 데이터

    차분된 데이터의 시각화와 Dickey-Fuller 테스트 결과를 출력합니다.
    """
    differenced_data = df.diff().dropna()

    # 차분 데이터 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(differenced_data, label='Differenced', color='red')
    plt.title('Differencing')
    plt.legend()
    plt.show()

    # Dickey-Fuller Test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(differenced_data.squeeze(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    print(dfoutput)

    # 정상성 여부 출력
    if dftest[1] <= 0.05:
        print('데이터는 정상성을 가집니다.')
    else:
        print('데이터는 정상성을 가지지 않습니다.')

    return differenced_data

# ACF 및 PACF 플롯 함수
def plot_acf_pacf(df, lag):
    """
    시계열 데이터의 ACF와 PACF 플롯을 그리는 함수입니다.

    Args:
    - df (pd.Series): ACF와 PACF를 그릴 시계열 데이터
    - lag (int): ACF 및 PACF를 계산할 최대 랙 수
    ACF와 PACF 그래프를 출력합니다.
    """
    # 두 그래프를 한 화면에 나란히 그리기
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # ACF 플롯 (첫 번째 축)
    plot_acf(df, lags=lag, ax=ax[0])

    # PACF 플롯 (두 번째 축)
    plot_pacf(df, lags=lag, ax=ax[1])

    # 플롯을 보여줌
    plt.tight_layout()
    plt.show()


# Train/Test Split 함수
def train_test_split(df):
    """
    시계열 데이터를 학습 및 테스트 데이터로 분할하는 함수입니다.

    Args:
    - df (pd.Series): 분할할 시계열 데이터

    Returns:
    - tuple: (train, test)로 분할된 데이터
    """
    train_size = int(len(df) * 0.8)
    train = df[:train_size]  # 학습 데이터
    test = df[train_size:]  # 테스트 데이터
    return train, test

# SARIMA 예측 함수
def sarima_forecast_plot(df, p, d, q, P, D, Q, m):
    """
    SARIMA 모델을 사용하여 시계열 데이터를 예측하는 함수입니다.

    Args:
    - df (pd.Series): 예측할 시계열 데이터
    - p, d, q (int): 비계절적 ARIMA 파라미터
    - P, D, Q, m (int): 계절적 ARIMA 파라미터

    Returns:
    - tuple: (forecast_df, model_summary)로 예측 결과와 모델 요약 정보를 반환

    학습, 예측, 평가 및 시각화를 수행합니다.
    """
    train, test = train_test_split(df)
    order = (p,d,q)  # p, d, q 값 설정
    seasonal_order = (P,D,Q,m)  # P, D, Q, m 값 설정
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # 4. 테스트 기간 예측 수행
    forecast = model_fit.get_forecast(steps=len(test))
    forecast_df = forecast.conf_int()
    forecast_df['yhat'] = forecast.predicted_mean
    forecast_df.index = test.index

    # 5. 테스트 데이터에 대한 성능 평가
    # 예측 성능 평가 지표 계산
    actual = test.values
    predicted = forecast_df['yhat'].values

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print(f"MAE (Mean Absolute Error): {mae}")
    print(f"RMSE (Root Mean Squared Error): {rmse}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

    # 6. 예측 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df, label='Actual Data', color='blue')
    plt.plot(forecast_df.index, forecast_df['yhat'], label='Predicted Data', linestyle='--', color='orange', )
    plt.axvline(x=test.index[0], color='red', linestyle='--', label='Forecast Start')
    # plt.fill_between(forecast_df.index, forecast_df.iloc[:, 0], forecast_df.iloc[:, 1], color='lightgreen', alpha=0.4)
    plt.title('Forecast with SARIMA')
    plt.xlabel('Date')
    plt.legend()
    plt.grid()
    plt.show()

    return forecast_df, model_fit.summary()


# SARIMA 최적 파라미터 찾기 함수
def sarima_grid_search(df, sarima_param_grid):
    """
    SARIMA 모델의 최적 파라미터를 찾기 위한 Grid Search 함수입니다.

    Args:
    - sarima_param_grid (list): SARIMA 파라미터 조합 리스트

    Returns:
    - dict: 최적의 SARIMA 파라미터 (order 및 seasonal_order)

    최적의 SARIMA 파라미터와 해당 RMSE 값을 출력합니다.
    """
    train, test = train_test_split(df)
    best_params = None
    best_rmse = float('inf')

    # Grid Search 수행
    for params in sarima_param_grid:
        try:
            # SARIMA 모델 생성 시 파라미터 분할
            order = (params[0], params[1], params[2])  # (p, d, q)
            seasonal_order = (params[3], params[4], params[5], params[6])  # (P, D, Q, s)
            
            # SARIMA 모델 생성 및 학습
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            
            # 테스트 기간 예측 수행
            forecast = model_fit.get_forecast(steps=len(test))
            predicted = forecast.predicted_mean
            actual = test.values
            
            # RMSE 계산
            rmse = np.sqrt(mean_squared_error(actual, predicted))

            # 최적 파라미터 갱신
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'order': order, 'seasonal_order': seasonal_order}
                
        except Exception as e:
            print(f"Error with parameters {params}: {e}")

    print(f"Best Parameters: {best_params}")
    print(f"Best RMSE: {best_rmse}")

    return best_params


# 최적 파라미터를 사용한 SARIMA 모델 학습 및 예측 함수
def train_sarima_with_best_params(df, best_params):
    """
    최적 파라미터를 사용하여 SARIMA 모델을 학습하고 예측하는 함수입니다.

    Args:
    - df (pd.Series): 예측할 시계열 데이터
    - best_params (dict): 최적의 SARIMA 파라미터 (order 및 seasonal_order)

    Returns:
    - tuple: (forecast_df, model_summary)로 예측 결과와 모델 요약 정보를 반환

    학습, 예측, 평가 및 시각화를 수행합니다.
    """
    train, test = train_test_split(df)
    # 5. 최적 파라미터를 사용한 SARIMA 모델 생성 및 학습
    model = SARIMAX(train, order=best_params['order'], seasonal_order=best_params['seasonal_order'])
    model_fit = model.fit(disp=False)

    # 6. 테스트 기간 예측 수행
    forecast = model_fit.get_forecast(steps=len(test))
    forecast_df = forecast.conf_int()
    forecast_df['yhat'] = forecast.predicted_mean
    forecast_df.index = test.index

    # 7. 테스트 데이터에 대한 성능 평가
    # 예측 성능 평가 지표 계산
    actual = test.values
    predicted = forecast_df['yhat'].values

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print(f"MAE (Mean Absolute Error): {mae}")
    print(f"RMSE (Root Mean Squared Error): {rmse}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

    # 8. 예측 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df, label='Actual Data', color='blue')
    plt.plot(forecast_df.index, forecast_df['yhat'], label='Predicted Data', linestyle='--', color='orange')
    plt.axvline(x=test.index[0], color='red', linestyle='--', label='Forecast Start')
    # plt.fill_between(forecast_df.index, forecast_df.iloc[:, 0], forecast_df.iloc[:, 1], color='lightgreen', alpha=0.4)
    plt.title('Forecast with SARIMA')
    plt.xlabel('Date')
    plt.legend()
    plt.grid()
    plt.show()

    return forecast_df ,model_fit.summary()


# SARIMA를 사용하여 2024년 상반기 예측 함수
def sarima_forecast_2024(df, best_params, s, f, c):
    """
    SARIMA 모델을 사용하여 2024년 상반기 예측을 수행하는 함수입니다.

    Args:
    - df (pd.Series): 예측할 시계열 데이터
    - s (int): 예측할 기간 (개월 수)
    - f (str): 날짜 주기 ('M' - 월 단위 등)
    - c (str): 그래프 선 색상

    Returns:
    - pd.DataFrame: 예측 결과 데이터프레임 (yhat, lower, upper)
    """
        # 5. 최적 파라미터를 사용한 SARIMA 모델 생성 및 학습
    model = SARIMAX(df, order=best_params['order'], seasonal_order=best_params['seasonal_order'])
    model_fit = model.fit(disp=False)

    # 6. 미래 기간 예측 수행 (2024년 상반기 6개월 예측)
    forecast = model_fit.get_forecast(steps=s)
    forecast_mean = forecast.predicted_mean  # 예측된 평균값
    forecast_conf = forecast.conf_int()  # 예측 구간 (상한과 하한)

    # 새로운 데이터프레임에 예측값과 예측 구간 저장
    forecast_df = pd.DataFrame({
        'yhat': forecast_mean,
        'lower': forecast_conf.iloc[:, 0],
        'upper': forecast_conf.iloc[:, 1]
    })
    forecast_df.index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=s, freq=f)
    
    # 8. 예측 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_df.index, forecast_df['yhat'],color=c)
    # plt.axvline(x=df_price['기준일'].iloc[-1], color='red', linestyle='--', label='Forecast Start')
    # plt.fill_between(forecast_df.index, forecast_df.iloc[:, 0], forecast_df.iloc[:, 1], color='lightgreen', alpha=0.4)
    plt.title('Forecast for 2024 with SARIMA')
    plt.xlabel('Date')
    plt.legend()
    plt.grid()
    plt.show()

    return forecast_df
