# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:18:55 2021

@author: 송호현
"""
import pandas as pd
import konlpy
from konlpy.tag import Kkma, Komoran, Okt, Hannanum
import re
import json

import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from konlpy.tag import Okt

train_data = pd.read_csv('news.csv')
train_data.dropna(inplace=True)
train_data = train_data.drop(['날짜', '언론사', '링크'], axis=1)
print(train_data)
train_data['기사제목'] = train_data['기사제목'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

# # 불용문 파일 일기
# worddf = pd.read_csv('stopwords.csv', header=None, encoding='utf-8')
# # 불용문파일 리스트에 저장
# sw = []
# for i in range(0, len(worddf)):
#     word = worddf[0][i]
#     sw.append(word)

stopwords = ['에도','의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# 형태소 분석기 OKT를 사용한 토큰화 작업 (다소 시간 소요)
okt = Okt()
def word_tokenization(text):
    stop_words = stopwords
    return [word for word in okt.morphs(text) if word not in stopwords and len(word) > 1 ]


data = train_data['기사제목'].apply((lambda x: word_tokenization(x)))
print(data)

# data 정보를 토큰화 하기
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()  # 토큰화 객체 생성
tokenizer.fit_on_texts(data)  # data값을 토큰화 하기
list(tokenizer.word_index.items())[:10]
print("총 단어 갯수 : ", len(tokenizer.word_index))  # 102194
# tokenizer.word_counts : 각 단어의 건수
list(tokenizer.word_counts.items())[:20]
# 5회 이상 사용된 단어만 추출하기
cnt = 0
for x in tokenizer.word_counts.values():
    if x >= 5:
        cnt += 1
print('5회 이상 사용된 단어 : ', cnt)
list(tokenizer.word_counts.values())
type(tokenizer.word_counts)

for key, value in tokenizer.word_counts.items():
    if value >= 5:
        print(key)

# with open('neg_pol_word.txt', encoding='utf-8') as neg:
#     negative = neg.readlines()
#
# negative = [neg.replace('\n', '') for neg in negative]
#
# with open('pos_pol_word.txt', encoding='utf-8') as pos:
#     positive = pos.readlines()
#
# positive = [pos.replace('\n', '') for pos in positive]

from tqdm import tqdm
import re
import pandas as pd

labels = []
keyword_df = []  # 긍부정 분석하기 위한 리스트
keyword_dic = {}  # word cloud 만들기 위한 딕셔너

for key, value in tokenizer.word_counts.items():
    if value >= 3:
        keyword_df.append(key)
        keyword_dic[key] = value

title_data = keyword_df
title_data

keyword_dic

# for title in tqdm(title_data):
#     clean_title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》]', '', title)
#     negative_flag = False
#     label = 0
#     for i in range(len(negative)):
#         if negative[i] in clean_title:
#             label = -1
#             negative_flag = True
#             print('negative 비교단어 : ', negative[i], 'clean_title : ', clean_title)
#             break
#     if negative_flag == False:
#         for i in range(len(positive)):
#             if positive[i] in clean_title:
#                 label = 1
#                 print('positive 비교단어 : ', positive[i], 'clean_title : ', clean_title)
#                 break
#
#     labels.append(label)

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

frequency = keyword_dic

# 워드 클라우드의 기본 설정
fp = 'MaruBuri-Regular.ttf'
wc = WordCloud(background_color='white', max_words=30, width=900, height=700, font_path=fp)
plt.figure(figsize=(15, 15))
wc = wc.generate_from_frequencies(frequency)
plt.imshow(wc.recolor(colormap='hsv'))
plt.axis('off')
plt.savefig('wordcloud_final.jpg')
plt.show()

















