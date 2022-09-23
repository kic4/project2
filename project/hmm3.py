#hmm lstm예측
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
#"Malgun Gothic " 폰트 설정
plt.rc('font', family='Malgun Gothic')


df = pd.DataFrame()
for page in range(1,21):
    url = 'http://finance.naver.com/item/sise_day.nhn?code=011200'
    url = '{url}&page={page}'.format(url=url, page=page)
    print(url)
    df = df.append(pd.read_html(requests.get(url, headers={'User-agent': 'Mozilla/5.0'}).text))
print(df.info())

df = df[['날짜', '시가', '고가', '저가', '종가', '거래량']]
df = df.dropna()
df = df.rename(columns={'날짜': 'date', '시가':'open', '고가':'high', '저가':'low','종가': 'close', '거래량':'volume'})
df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(int)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['date'], ascending=True)
df.reset_index(inplace=True)
print(df)

# # 거래량
# plt.figure(figsize=(16,9))
# sns.lineplot(y=df['volume'],x=df['date'])
# plt.xlabel('time')
# plt.ylabel('volume')
# plt.show()

# # 거래대금
# plt.figure(figsize=(16,9))
# sns.lineplot(y=df['trade_volume'],x=df['date'])
# plt.xlabel('time')
# plt.ylabel('trade_volume')
# plt.show()

# # 주가
# plt.figure(figsize=(16,9))
# sns.lineplot(y=df['close'],x=df['date'])
# plt.xlabel('time')
# plt.ylabel('close')
# plt.show()

# 데이터 정규화
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scale_cols=['open','high','low','close','volume']
df_scaled=scaler.fit_transform(df[scale_cols])

df_scaled=pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols
print(df_scaled)

# 시계열 데이터의 데이터셋 분리
test_size=100
window_size=10

train=df_scaled[:-test_size]
test=df_scaled[-test_size:]

def make_dataset(data, label, window_size) :
    feature_list=[]
    label_list=[]
    for i in range(len(data)-window_size) :
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

feature_cols=['open','high','low','volume']
label_cols=['close']

train_feature=train[feature_cols]
train_label=train[label_cols]
train_feature, train_label = make_dataset(train_feature, train_label, window_size)

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid=train_test_split(train_feature, train_label, test_size=0.2)

test_feature=test[feature_cols]
test_label=test[label_cols]

test_feature, test_label=make_dataset(test_feature, test_label, window_size)


# 모형 학습
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 모형 만들기
model=Sequential()
model.add(LSTM(128,input_shape=(train_feature.shape[1],train_feature.shape[2]),activation='relu',
               return_sequences=False))
model.add(Dense(1))

print(model.summary())

# 모형 학습
import os

model.compile(loss='mean_squared_error', optimizer='adam')
early_stop=EarlyStopping(monitor='val_loss',patience=5)

model_path='model'
filename=os.path.join(model_path,'tmp_checkpoint.h5')
checkpoint=ModelCheckpoint(filename, monitor='val_loss',verbose=1,save_best_only=True,mode='auto')
history=model.fit(x_train,y_train,epochs=200,batch_size=128,validation_data=(x_valid,y_valid),
                  callbacks=[early_stop,checkpoint])

model.load_weights(filename)
pred=model.predict(test_feature)

pred.shape

plt.figure(figsize=(12, 9))
plt.plot(test_label, label = 'actual')
plt.plot(pred, label = 'prediction')
plt.legend()
plt.title("LSTM 모델")
plt.show()

# r2-score, RMSE 값을 출력하기
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from math import sqrt
r2=r2_score(test_label,pred)
print("r2 :", r2)
rmse=sqrt(mean_squared_error(test_label,pred))
print("rmse :", rmse)
msle=mean_squared_log_error(test_label,pred)
print("msle :", msle)
rmsle=msle**0.5
print("rmsle :", rmsle)
