import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from konlpy.tag import Okt

train_data = pd.read_csv('news.csv')
train_data.dropna(inplace=True)
train_data = train_data.drop(['날짜', '언론사', '링크'], axis=1)
print(train_data)
train_data['기사제목'] = train_data['기사제목'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# 형태소 분석기 OKT를 사용한 토큰화 작업 (다소 시간 소요)
okt = Okt()
tokenized_data = []
for sentence in train_data['기사제목']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    tokenized_data.append(temp_X)

print(tokenized_data[:5]) # 상위 5개 출력

from gensim.models import Word2Vec
model = Word2Vec(sentences = tokenized_data, vector_size= 100, window = 5, min_count = 5, workers = 4, sg = 0)
print(model.wv.vectors.shape)

