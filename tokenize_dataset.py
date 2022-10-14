import multiprocessing
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf

sns.set_style("darkgrid")

data = np.array(pd.read_csv('horoscopes.csv'))


def process_text(dataset):
    tokens = []
    for line in dataset:
        new_token = tf.keras.preprocessing.text.text_to_word_sequence(input_text=line[2],
                                                                      filters='!"#$%&amp;()*+,-./:;&lt;=>?@[\\]^_`{'
                                                                              '|}~\t\n\ufeff',
                                                                      lower=True, split=' ')
        tokens.append(new_token)
    return tokens


def create_model(dataset):
    tokens = process_text(dataset)

    # Создадим модель
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=1,
                         window=5,
                         # size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores - 2)

    # Загрузим словарь
    t = time()
    w2v_model.build_vocab(tokens, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    # Обучим модель на гороскопах и сохраним веса
    w2v_model.train(tokens, total_examples=w2v_model.corpus_count, epochs=400, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    w2v_model.save("weights/word2vec.model")

    # Создадим новую модель на основе сохраненных весов
    w2v_model = Word2Vec.load("weights/word2vec.model")
    return w2v_model


def tsnescatterplot(model, word, list_names):
    # инициализируем массив слов для вывода
    arrays = np.empty((0, 300), dtype='f')

    word_labels = [word]
    color_list = ['red']

    # добавим вектор опорного слова в массив вывода
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

    # найдем наиболее похожие слова по отношению к опорному
    close_words = model.wv.most_similar([word])

    # добавим вектора похожих слов в массив вывода
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # добавим вектора интересующих слов в массив вывода
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # с помощью метода главных компонент уменьшим размероность массива вывода
    reduc = PCA(n_components=12).fit_transform(arrays)

    # сделаем вывод чисел с плав. зап. как вывод чисел с фикс. зап.
    np.set_printoptions(suppress=True)

    # получим умную 2-х мерную матрицу вывода из 12-ти мерной
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    # создадим дф для вывода
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)

    # проставляем точки
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                  }
                     )

    # проставляем сами слова
    for line in range(0, df.shape[0]):
        p1.text(df["x"][line],
                df['y'][line],
                '  ' + df["words"][line].title(),
                horizontalalignment='left',
                verticalalignment='bottom', size='medium',
                color=df['color'][line],
                weight='normal'
                ).set_size(15)

    plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
    plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

    plt.title('t-SNE visualization for {}'.format(word.title()))


# model = create_model(data)
model = Word2Vec.load("weights/word2vec.model")
tsnescatterplot(model, 'июнь', ['чары', 'неудача', 'удача', 'собака', 'труд', 'кредит', 'счастье'])
