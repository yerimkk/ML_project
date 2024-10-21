from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import itertools

# 한국어 폰트 설정 (맑은 고딕 사용)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows에서 '맑은 고딕' 폰트 경로
fontprop = fm.FontProperties(fname=font_path)

# 그래프 시각화 시, 폰트 설정
plt.rc('font', family=fontprop.get_name())
plt.rcParams['axes.unicode_minus'] = False

# 시계열 분해 함수 정의
def seasonal_decompose_graph(df, col_name, p):
    """
    시계열 데이터를 additive 및 multiplicative 모델로 분해하고 그래프를 출력하는 함수.
    
    Args:
    - df (DataFrame): 시계열 데이터가 포함된 데이터프레임.
    - col_name (str): 분해할 시계열 열 이름.
    - p (int): 주기 (예: 월별 데이터일 경우, 주기=12).
    """
    # 시계열 분해 (additive 모델 사용)
    print("Additive 모델")
    result = seasonal_decompose(df[col_name], model='additive', period=p)
    result.plot()
    plt.show()

    # 시계열 분해 (multiplicative 모델 사용)
    print("Multiplicative 모델")
    result = seasonal_decompose(df[col_name], model='multiplicative', period=p)
    result.plot()
    plt.show()

# Train/Test Split 함수 정의
def split_train_test(df, col_name):
    """
    시계열 데이터를 학습 데이터와 테스트 데이터로 분할하는 함수.
    
    Args:
    - df (DataFrame): 시계열 데이터가 포함된 데이터프레임.
    - col_name (str): 분할할 대상 열 이름.
    
    Returns:
    - df_tmp (DataFrame): '기준일'과 대상 열을 'ds'와 'y'로 변경한 데이터프레임.
    - train (DataFrame): 학습 데이터 (80%).
    - test (DataFrame): 테스트 데이터 (20%).
    """
    df_tmp = df[['기준일', col_name]].rename(columns={'기준일': 'ds', col_name: 'y'})

    # Train/Test Split (80% train, 20% test)
    train_size = int(len(df_tmp) * 0.8)
    train = df_tmp[:train_size]  # 학습 데이터
    test = df_tmp[train_size:]  # 테스트 데이터

    return df_tmp, train, test

# Train/Test Split 함수 정의 (생산/소비 데이터용)
def split_train_test_production_consumption(production_consumption, col_name):
    """
    생산/소비 데이터를 학습 데이터와 테스트 데이터로 분할하는 함수.
    
    Args:
    - production_consumption (DataFrame): 생산/소비 데이터가 포함된 데이터프레임.
    - col_name (str): 분할할 대상 열 이름.
    
    Returns:
    - df_tmp (DataFrame): '연도'와 대상 열을 'ds'와 'y'로 변경한 데이터프레임.
    - train (DataFrame): 학습 데이터 (80%).
    - test (DataFrame): 테스트 데이터 (20%).
    """
    df_tmp = production_consumption[['연도', col_name]].rename(columns={'연도': 'ds', col_name: 'y'})
    df_tmp['ds'] = pd.to_datetime(df_tmp['ds'], format='%Y')

    # Train/Test Split (80% train, 20% test)
    train_size = int(len(df_tmp) * 0.8)
    train = df_tmp[:train_size]  # 학습 데이터
    test = df_tmp[train_size:]  # 테스트 데이터

    return df_tmp, train, test

# Prophet 예측, 성능 지표, 예측값 시각화 함수 정의
def prophet_metrix_plot(train, test, f):
    """
    Prophet 모델을 사용하여 예측, 성능 지표 계산 및 예측값 시각화를 수행하는 함수.
    
    Args:
    - train (DataFrame): 학습 데이터.
    - test (DataFrame): 테스트 데이터.
    - f (str): 예측 주기 (예: 'M' for monthly).
    """
    model = Prophet()
    model.fit(train)

    # 테스트 기간 예측 수행
    future = model.make_future_dataframe(periods=len(test), freq=f)
    forecast = model.predict(future)

    # 테스트 기간에 해당하는 예측 값 가져오기
    test_forecast = forecast[-len(test):]

    # 실제 값과 예측 값 준비
    actual = test['y'].values
    predicted = test_forecast['yhat'].values

    # 성능 지표 계산
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print(f"MAE (Mean Absolute Error): {mae}")
    print(f"RMSE (Root Mean Squared Error): {rmse}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

    # 예측 결과 시각화
    model.plot(forecast)
    plt.show()

    # 변화점 및 계절성 플롯
    model.plot_components(forecast)
    plt.show()

    # 예측값 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(test['ds'], actual, label='Actual (Test Data)', color='blue')
    plt.plot(test['ds'], predicted, label='Predicted (Prophet)', color='orange', linestyle='--')
    plt.title('Prediction vs Actual')
    plt.xlabel('Date')
    plt.ylabel('')
    plt.legend()
    plt.grid()
    plt.show()

# Prophet 하이퍼파라미터 튜닝 함수 정의
def tune_prophet_parameters(param_grid, train, test, f):
    """
    Prophet 모델의 하이퍼파라미터를 튜닝하기 위한 함수 (Grid Search 사용).
    
    Args:
    - param_grid (dict): 하이퍼파라미터 그리드 딕셔너리.
    - train (DataFrame): 학습 데이터.
    - test (DataFrame): 테스트 데이터.
    - f (str): 예측 주기 (예: 'M' for monthly).
    
    Returns:
    - best_params (dict): 최적의 하이퍼파라미터 딕셔너리.
    """
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    best_params = None
    best_rmse = float('inf')

    for params in all_params:
        model = Prophet(**params)
        model.fit(train)
        
        future = model.make_future_dataframe(periods=len(test), freq=f)
        forecast = model.predict(future)

        test_forecast = forecast[-len(test):]
        predicted = test_forecast['yhat'].values
        actual = test['y'].values
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        if rmse < best_rmse:
            best_params = params
            best_rmse = rmse

    print(f"Best Parameters: {best_params}")
    print(f"Best RMSE: {best_rmse}")
    return best_params

# 최적 파라미터를 사용한 Prophet 예측 및 시각화 함수 정의
def tune_prophet_parameters_plot(best_params, train, test, f):
    """
    최적의 하이퍼파라미터로 Prophet 모델을 학습하고 예측 및 시각화를 수행하는 함수.
    
    Args:
    - best_params (dict): 최적의 하이퍼파라미터 딕셔너리.
    - train (DataFrame): 학습 데이터.
    - test (DataFrame): 테스트 데이터.
    - f (str): 예측 주기 (예: 'M' for monthly).
    """
    model = Prophet(**best_params)
    model.fit(train)

    future = model.make_future_dataframe(periods=len(test), freq=f)
    forecast = model.predict(future)

    test_forecast = forecast[-len(test):]
    actual = test['y'].values
    predicted = test_forecast['yhat'].values

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print(f"MAE (Mean Absolute Error): {mae}")
    print(f"RMSE (Root Mean Squared Error): {rmse}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

    model.plot(forecast)
    plt.show()

    model.plot_components(forecast)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(test['ds'], actual, label='Actual (Test Data)', color='blue')
    plt.plot(test['ds'], predicted, label='Predicted (Prophet)', color='orange', linestyle='--')
    plt.title('Prediction vs Actual')
    plt.xlabel('Date')
    plt.ylabel('')
    plt.legend()
    plt.grid()
    plt.show()

# 2024년 예측 함수 정의
def predict_2024(df, p, f, c, best_params):
    """
    2024년 데이터를 예측하고 결과를 시각화하는 함수.
    
    Args:
    - df (DataFrame): 전체 학습 데이터.
    - p (int): 예측할 기간.
    - f (str): 예측 주기 (예: 'M' for monthly).
    - c (str): 시각화 색상.
    - best_params (dict): 최적의 하이퍼파라미터 딕셔너리.
    
    Returns:
    - test_forecast (DataFrame): 예측 결과 데이터프레임.
    """
    model = Prophet(**best_params)
    model.fit(df)

    future = model.make_future_dataframe(periods=p, freq=f)
    forecast = model.predict(future)

    test_forecast = forecast[-p:]

    model.plot(forecast)
    plt.show()

    model.plot_components(forecast)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(forecast['ds'][-p:], forecast['yhat'][-p:], label='Forecasted Data', color=c)
    plt.title('')
    plt.xlabel('Date')
    plt.ylabel('')
    plt.legend()
    plt.grid()
    plt.show()

    return test_forecast
