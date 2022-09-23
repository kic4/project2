from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

texts = ['You are the Best Thing', 'You are the Nice']
#Tokenizer = 10 : 10개의 토근으로 분리.
#oov_token : 기존분석된 토근이 없는 경우 대체되는 값
tokenizer = Tokenizer(num_words=10, oov_token='<OOV>')
# texts 문장을 토큰화 실행.
tokenizer.fit_on_texts(texts)
#texts_to_sequences : 토큰값을 정수인덱스로 변환
sequences = tokenizer.texts_to_sequences(texts)
#sequences_to_matrix : 이진 형태 인코딩
binary_results = tokenizer.sequences_to_matrix(sequences, mode='binary')
#토큰화된 결과값. 토큰값과 인덱스값을 출력.
print(tokenizer.word_index)
print(sequences)
print(binary_results)

test_text = ['You are the One']
test_seq = tokenizer.texts_to_sequences(test_text)
print(test_seq)

'''
imdb 데이터 셋
-영화리뷰에 대한 데이터 5만개
-50%씩 긍정리뷰, 부정리뷰
-전처리가 완료상태. -> 내용이 숫자로 변환됨.
'''
from tensorflow.keras.datasets import imdb
num_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
print(X_train.shape, X_test.shape)
print(X_train[0])
print(len(X_train[0]))
print(len(X_train[1]))

imdb_get_word_index ={}
for key, value in imdb.get_word_index().items():
    imdb_get_word_index[value] = key
for i in range(1, 6):
    print('{} 번쨰로 가장 많이 쓰인 단어 = {}'.format(i, imdb_get_word_index[i]))

#훈련데이터의 문장 길이의 평균과 중간값, 최대값, 최소값 출력
import numpy as np
#comprehension
lengths = np.array([len(x) for x in X_train])
lengths[:10]
#평균
np.mean(lengths)
np.median(lengths)

#분석을 위해서 데이터의 길이를 동일하게 처리해야함
#패딩 작업이 필요함 : 데이터의 길이가 지정 길이 보다 적으면 0으로 채움
#           데이터의 길이가 지정 길이보다 크면 지정 길이로 짤라냄
#지정 길이
max_len=500
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_X_train = pad_sequences(X_train, maxlen=max_len, padding='pre')

import tensorflow as tf
import pandas as pd
train_file = tf.keras.utils.get_file('ratings_train.txt', \
        origin='https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt', \
            extract=True)
train= pd.read_csv(train_file, sep='\t')
train.head()
print("train shape: ", train.shape)
train.info()