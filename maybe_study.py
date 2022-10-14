from tensorflow.keras.layers import Activation, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import concatenate, Reshape, Add, LSTM, Multiply, Embedding, LSTM, GRU, BatchNormalization

from tensorflow.keras.layers import Bidirectional, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import re

import pymorphy2

from tensorflow.keras.initializers import Constant
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences, TimeseriesGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import collections

text = np.array(pd.read_csv('horoscopes.csv'))
predictions = text[:2]


def process_text(data):
    tokens = []
    for line in data:
        new_token = text_to_word_sequence(line[2], filters='!"#$%&amp;()*+,-./:;&lt;=>?@[\\]^_`{|}~\t\n\ufeff',
                                          lower=True, split=' ')
        tokens.append(new_token)
    return tokens


text_tokens = process_text(text)
print(text_tokens[0])

num_words = 20000  # 84267
tokenizer = Tokenizer(
    num_words=num_words,
    filters='!"#$%&amp;()*+,-—./:;&lt;=>?@[\\]^_`{|}~\t\n\xa0\ufeff',
    lower=True,
    split=' ',
    char_level=False)

tokenizer.fit_on_texts(text_tokens)
sequences = tokenizer.texts_to_sequences(text_tokens)

max_len = 0  # максимальная длина предложения

for seq in sequences:
    l = len(seq)
    if l > max_len:
        max_len = l

vocab_len = len(tokenizer.word_index)
embedding_dim = 300  # длина эмбеддинга слова

print('максимальное количество слов в предложении ', max_len)
print('длина словаря ', vocab_len)
print('длина эмбеддинга ', embedding_dim)

sequences = pad_sequences(sequences=sequences, maxlen=max_len, padding='post')
print('текст предложения\n', sequences[1][:50])
print('размерность всего массива', sequences.shape)

model_w2v = KeyedVectors.load_word2vec_format('weights/araneum_upos_skipgram_300_2_2018.vec.gz')
w2v = Word2Vec.load("weights/word2vec.model")

conv_pos = {'ADJF': 'ADJ', 'ADJS': 'ADJ', 'ADV': 'ADV', 'NOUN': 'NOUN',
            'VERB': 'VERB', 'PRTF': 'ADJ', 'PRTS': 'ADJ', 'GRND': 'VERB'}

tmp_dict = {}  # Кеш значимых слов.
nones = {}  # Кеш незначимых слов.

morph = pymorphy2.MorphAnalyzer()


# Фильтруем по части речи и возвращаем только начальную форму.
def normalizePymorphy(text, need_pos=True):
    tokens = re.findall('[A-Za-zА-Яа-яЁё]+\-[A-Za-zА-Яа-яЁё]+|[A-Za-zА-Яа-яЁё]+', text)
    words = []
    for t in tokens:
        # Если токен уже был закеширован, быстро возьмем результат из него.
        if t in tmp_dict.keys():
            words.append(tmp_dict[t])
        # Аналогично, если он в кеше незначимых слов.
        elif t in nones.keys():
            pass
        # Слово еще не встретилось, будем проводить медленный морфологический анализ.
        else:
            pv = morph.parse(t)
            if pv[0].tag.POS is not None:
                if pv[0].tag.POS in conv_pos.keys():
                    if need_pos:
                        word = pv[0].normal_form + "_" + conv_pos[pv[0].tag.POS]
                    else:
                        word = pv[0].normal_form
                    # Отправляем слово в результат, ...
                    words.append(word)
                    # ... и кешируем результат его разбора.
                    tmp_dict[t] = word
                else:
                    # Для незначимых слов можно даже ничего не хранить. Лишь бы потом не обращаться к морфологии.
                    nones[t] = ""

    return words


index2word_set = set(model_w2v.index_to_key)


def text_to_vec(text):
    text_vec = np.zeros((model_w2v.vector_size,), dtype="float32")
    n_words = 0

    for word in normalizePymorphy(text):
        if word in index2word_set:
            n_words = n_words + 1
            text_vec = np.add(text_vec, model_w2v[word])

    if n_words != 0:
        text_vec /= n_words
    return text_vec


w2v_vectors = [text_to_vec(txt) for txt in predictions]
embedding_matrix = w2v.wv.vectors

scaler = MinMaxScaler((-1, 1))
print('до нормализации\n', embedding_matrix[458][:10])

embedding_matrix = scaler.fit_transform(embedding_matrix)
print('после нормализации\n', embedding_matrix[458][:10])


def word2idx(word):
    return w2v.wv.vocab[word].index


def idx2word(idx):
    return w2v.wv.index_to_key[idx]


window_step = 5
N = 500000
step = 1
count = 0

x = []
y = []

for line in sequences:
    if count >= N:
        break

    for j in range(0, len(line) - step - window_step, step):
        temp = []
        for i in range(window_step):
            temp.append(line[j + i])

        x.append(temp)
        y.append(embedding_matrix[line[j + window_step]])

x = np.array(x)
y = np.array(y)

print('x shape ', x.shape)
print('y shape ', y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, shuffle=False)  # , random_state = 42)

model = Sequential()

model.add(Embedding(input_dim=vocab_len,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    trainable=False
                    ))
model.add(Dense(300, input_shape=(window_step, embedding_dim)))
model.add(LeakyReLU(0.2))

# modelGRU.add(Embedding(num_words, embedding_size))
# modelGRU.add(SpatialDropout1D(0.2))

model.add(LSTM(128, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(BatchNormalization())

model.add(LSTM(256, return_sequences=True))
model.add(GRU(512, return_sequences=True))
model.add(BatchNormalization())

model.add(LSTM(512, return_sequences=True))
model.add(GRU(1024))
model.add(BatchNormalization())

# modelGRU.add(Bidirectional(LSTM(128)))
# modelGRU.add(LSTM(8,return_sequences=True ))
# model.add(LSTM(128))

model.add(Dense(256))
model.add(LeakyReLU(0.2))
model.add(Dropout(0.2))

model.add(Dense(64))
model.add(LeakyReLU(0.2))

model.add(Dense(embedding_dim, activation='tanh'))
# model.summary()
optimizer = RMSprop(0.0001)
# optimizer=Adam(0.0001)
model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
history = model.fit(x, y, batch_size=3000, epochs=50, validation_split=0.1, verbose=1)

model.save_weights('modelGRU.h5')
model.load_weights_weights('modelGRU.h5')
emb2indx = Sequential()

emb2indx.add(Dense(256, input_shape=(300,)))
emb2indx.add(LeakyReLU(0.2))

emb2indx.add(Dense(1024))
emb2indx.add(LeakyReLU(0.2))

emb2indx.add(Dense(vocab_len))
emb2indx.add(Activation('softmax'))

emb2indx.summary()

embedding_matrix_label = np.arange(0, vocab_len)

x_train, _, y_train, _ = train_test_split(embedding_matrix,
                                          embedding_matrix_label,
                                          test_size=0.01,
                                          shuffle=True)  # , random_state = 42)

optimizer = RMSprop(0.01)
optimizer = Adam(0.01)
emb2indx.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["categorical_accuracy"])
history2 = emb2indx.fit(x_train, y_train, batch_size=512, epochs=20, validation_split=0.1, verbose=1)

plt.figure(figsize=(10, 10))
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.legend(['loss', 'vall_loss'], fontsize=15)
plt.xlabel('epochs', fontsize=15)
plt.show()

words = ['Сегодня раки сделали весам хороший']
start_words = np.zeros((1, 5, embedding_dim))

sents = tokenizer.texts_to_sequences(words)

for i, sent in enumerate(sents[0]):
    embedding = embedding_matrix[sent].reshape((1, -1))
    start_words[0, i] = scaler.inverse_transform(embedding)

print(start_words[0, :, :20])
