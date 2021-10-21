# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:52:18 2021

@author: 송호현
"""

# 시계열 데이터 분석

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Alteogen.csv',header=0)
df.info()
df=df.astype({'date':'str'})
df=df[df['date'].str[:4] == '2021']
df=df.sort_values(by='date', ascending=True)
df.head()
# date 컬럼을 시계열 피처로 변경하기
# date 컬럼을 문자열 형태 -> 날짜형으로 변경
df['date']=pd.to_datetime(df['date'])
df.info()
df.describe()

# date 컬럼을 인덱스로 변경하기
df.set_index('date',inplace=True)
df.head()
alteogen_df=df[['close']]
alteogen_df.info()

alteogen_df.plot()
plt.show()

from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(alteogen_df.close.values, order=(2,1,2))

# 학습하기
model_fit=model.fit(trend='c', full_output=True, disp=True)
fig=model_fit.plot_predict() # 예측 결과를 그래프로 출력

# resid : 잔차정보
residuals=pd.DataFrame(model_fit.resid)
residuals.plot() # 두번째 그래프

# 예측데이터
forecast_month=model_fit.forecast(steps=30) # 30일 정보를 예측
forecast_month

'''
1번 배열 : 예측값. 5일치 예측값
2번 배열 : 표준 오차
3번 배열 : [예측 하한값, 예측 상한값]
'''
# 실데이터
alteogen_test_df=pd.read_excel('Alteogen_test.xlsx',names=['ds','y'],header=0)
alteogen_test_df

# 30일치 예측데이터
test_y=alteogen_test_df.y.values
pred_y=forecast_month[0].tolist()
pred_y_lower=[] # 예측 하한값
pred_y_upper=[] # 예측 상한값
for low_up in forecast_month[2] :
    pred_y_lower.append(low_up[0])
    pred_y_upper.append(low_up[1])

plt.plot(pred_y, color="gold") # 예측값
plt.plot(test_y, color='green') # 실제값
plt.plot(pred_y_lower, color='red') # 예측 하한값
plt.plot(pred_y_upper, color='blue') # 예측 상한값
plt.show()

# r2-score, RMSE 값을 출력하기
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from math import sqrt
r2=r2_score(test_y,pred_y)
r2
rmse=sqrt(mean_squared_error(test_y,pred_y))
rmse
msle=mean_squared_log_error(test_y,pred_y)
msle
rmsle=msle**0.5
rmsle
