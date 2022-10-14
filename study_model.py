import numpy as np
import multiprocessing
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from time import time
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score

from sklearn.cluster import DBSCAN
import seaborn as sns
from matplotlib.colors import ListedColormap
from numpy.random import choice
from umap import UMAP

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import re

import pymorphy2

sns.set_style("darkgrid")

text=np.array(pd.read_csv('horoscopes.csv'))
predictions=text[:,2]
model_w2v= KeyedVectors.load_word2vec_format('weights/araneum_upos_skipgram_300_2_2018.vec.gz')
signs=['aries','gemini','taurus','cancer','leo','virgo','libra','scorpio','sagittarius','capricorn','aquarius','pisces']
signs_r=['овен','близнецы','телец','рак','лев','дева','весы','скорпион','стрелец','козерог','водолей','рыбы']
w2v_model = Word2Vec.load("weights/word2vec.model")
news=pd.read_csv('horoscopes.csv')
print(news.columns)

elem=10
news=news[news.sign.str.contains(signs[elem])]
def processText(data):
    #
    #  разобьем тексты на слова, убрав мусор
    #
    tokens=[]
    for line in data:
        newToken=text_to_word_sequence(line,filters='!"#$%&amp;()*+,-./:;&lt;=>?@[\\]^_`{|}~\t\n\ufeff',
                                  lower=True,split=' ')
        tokens.append(newToken)
    return tokens
text=news.values[:,2]
tokens=processText(text)
print(tokens[1])

# Список значимых частей речи.
# Они нам потом понадобятся в немного другом виде. Так что сделаем словарь. чтобы два раза не вставать.
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
            if pv[0].tag.POS != None:
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

        #    if n_words != 0:
    #        text_vec /= n_words
    return text_vec


index2word_set = set(model_w2v.index_to_key)


def text_to_vec(text):
    text_vec = np.zeros((model_w2v.vector_size,), dtype="float32")
    n_words = 0

    for word in normalizePymorphy(text):
        if word in index2word_set:
            n_words = n_words + 1
            text_vec = np.add(text_vec, model_w2v[word])

        #    if n_words != 0:
    #        text_vec /= n_words
    return text_vec


print(predictions.shape)

w2v_vectors = [text_to_vec(txt) for txt in predictions]
avg_text=np.zeros((len(tokens),100))
for i,line in enumerate(tokens):
    temp=[w2v_model.wv[word] for word in line]
    mean=sum(temp)/len(temp)
    avg_text[i]=mean
umap_news=UMAP()
umaped_vct=umap_news.fit_transform(w2v_vectors)
N=15
fig, ax = plt.subplots(figsize=(N,N))
ax.scatter(umaped_vct[:, 0], umaped_vct[:, 1], edgecolor='b', s=3)
plt.show()