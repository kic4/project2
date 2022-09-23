#hmm fbprophet 예측
import pandas as pd
from fbprophet import Prophet
import matplotlib.font_manager as fm
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as plyo
import cufflinks

#"Malgun Gothic " 폰트 설정
plt.rc('font', family='Malgun Gothic')

df = pd.DataFrame()
for page in range(1,50):
    url = 'http://finance.naver.com/item/sise_day.nhn?code=011200'
    url = '{url}&page={page}'.format(url=url, page=page)
    print(url)
    df = df.append(pd.read_html(requests.get(url, headers={'User-agent': 'Mozilla/5.0'}).text))
print(df.info())

df = df[['날짜', '종가']]
df = df.dropna()

hmm_df = df.rename(columns={'날짜': 'ds', '종가': 'y'})
hmm_df['y'] = hmm_df['y'].astype(int)
hmm_df['ds'] = pd.to_datetime(hmm_df['ds'])
hmm_df = hmm_df.sort_values(by=['ds'], ascending=True)
print(hmm_df)

hmm_train_df = hmm_df[:460]
hmm_test_df = hmm_df[460:]

prophet = Prophet(seasonality_mode = 'multiplicative',
                 yearly_seasonality=True,
                 weekly_seasonality=True,
                 daily_seasonality=True,
                 changepoint_prior_scale=0.5)

prophet.fit(hmm_train_df)

future_data = prophet.make_future_dataframe(periods = 30, freq = 'd')
print(future_data.tail(10))
forecast_data = prophet.predict(future_data)
forecast_data[['ds','yhat', 'yhat_lower', 'yhat_upper']].head()

#학습결과 시각화, 트랜드 정보 시각화 그래프
prophet.plot(forecast_data)
prophet.plot_components(forecast_data)

#Testset 평가
plt.figure(figsize=(15, 10))
pred_fbprophet_y = forecast_data.yhat.values[-30:]
test_y = hmm_test_df.y.values
pred_y_lower = forecast_data.yhat_lower.values[-30:]
pred_y_upper = forecast_data.yhat_upper.values[-30:]

# 모델이 예측 그래프
plt.plot(pred_fbprophet_y, color = 'gold')
plt.plot(pred_y_lower, color = 'red')
plt.plot(pred_y_upper, color = 'blue')
plt.plot(test_y, color = 'green')
plt.legend(['예측값', '최저값', '최대값', '실제값'])
plt.title("값 비교")
plt.show()

# r2-score, RMSE 값을 출력하기
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from math import sqrt
r2=r2_score(test_y, pred_fbprophet_y)
print("r2 :", r2)
rmse=sqrt(mean_squared_error(test_y, pred_fbprophet_y))
print("rmse :", rmse)
msle=mean_squared_log_error(test_y, pred_fbprophet_y)
print("mlse :", msle)
rmsle=msle**0.5
print("rmsle:", rmsle)

#반응형 차트
fig = px.line(hmm_df, x='ds', y='y', title='HMM')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(step="all")
        ])
    )
)
print(fig.to_json())

hmm_df.set_index('ds', inplace=True)
fig2 = hmm_df.iplot(asFigure=True)
# fig2 = plyo.iplot(hmm_df.iplot(asFigure=True))
print(fig2.to_json())
fig2.show()


