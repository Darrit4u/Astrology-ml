import numpy as np
import pandas as pd
import pymorphy2

data = pd.read_csv('horoscopes.csv')

# Удалим пропуски и сверхкороткие строки
for i in range(len(data) - 1, -1, -1):
    text = str(data['text'][i])
    sign = str(data['sign'][i])
    date = str(data['date'][i])
    if len(sign) < 2:
        data = data.drop([i])
    if len(text) < 2:
        data = data.drop([i])
    if len(date) > 13:
        data = data.drop([i])
    if date[0] != "2":
        data = data.drop([i])

# Удалим из текста все, что не текст
char_not_text = ['.', ',', ')', '(', '»', '«', '"', ':', '!']

dataset = np.array(data)
for i in range(dataset.shape[0]):
    text = str(dataset[i][2])
    for char in char_not_text:
        text = text.replace(char, '')
    text = text.replace('-', ' ')
    text = text.lower()
    dataset[i][2] = text


# Создадим словарь из слов датасета, приведенных к именительному падежу
raw_words = set()
morph = pymorphy2.MorphAnalyzer()

# Собираем список всех слов
for i in range(dataset.shape[0]):
    text = str(dataset[i][2])
    text_chars = text.split(' ')
    for word in text_chars:
        raw_words.add(word)

# Сохраним множество слов
words_list = []
for word in raw_words:
    words_list.append({'word': word})

words_df = pd.DataFrame(words_list)
words_df.to_csv('words.csv')

# Посчитаем для каждого слова в каком предложении и сколько раз повторялось
words = pd.read_csv('words.csv')
words.head()
chars = np.array(words)

chars_set = set()
for i, word in enumerate(chars):
    if type(word[1]) is not str:
        print(f"{word[1]} is not str: type = {type(word[1])}\n")
        continue
    text = morph.parse(word[1])[0].normal_form
    chars_set.add(text)

signs_set = set()
for line in dataset:
    signs_set.add(line[1])
signs_list = [sign for sign in signs_set]
chars_list = [char for char in chars_set]
stat_data = np.zeros((len(chars_list), len(signs_list)))
for i, line in enumerate(dataset):
    print(f"{i}/{len(dataset)}\n")
    sign = line[1]
    text = str(line[2])
    text = text.split(' ')
    for word in text:
        if word == '':
            continue
        fixed = morph.parse(word)[0].normal_form
        x = chars_list.index(fixed)
        y = signs_list.index(sign)
        stat_data[x][y] += 1

print(stat_data)

np.save('parsedArrays/stat_data.npy', stat_data)
np.save('parsedArrays/signs_list.npy', signs_list)
np.save('parsedArrays/chars_list.npy', chars_list)
