import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

# Выведем полученные статистические данные на диаграмму
stat_data = np.load('parsedArrays/stat_data.npy')
stat_data_list = list(stat_data)
signs = np.load('parsedArrays/signs_list.npy')
chars_list = list(np.load('parsedArrays/chars_list.npy'))
signs_dict = {'aries': 0, 'gemini': 1, 'taurus': 2, 'cancer': 3, 'leo': 4, 'virgo': 5, 'libra': 6,
              'scorpio': 7, 'sagittarius': 8, 'capricorn': 9, 'aquarius': 10, 'pisces': 11}

sign = 'aries'  # можно тут поменять знак зодиака на другой, чтобы вывести соответсвующую диаграмму
groups = [i for i in range(len(chars_list))]
counts = np.array([num[signs[sign]] for num in stat_data])
counts_list = list(counts)
sum = counts.sum(axis=0)

counts.sort(axis=0)
N = 100
common_vals = counts[::-1]
common_vals = common_vals[:N]

common_indeces = []

for val in common_vals:
    key = counts_list.index(val)
    common_indeces.append(key)

common_words = []

for index in common_indeces:
    common_words.append(chars_list[index])

dpi = 150
fig = plt.figure(dpi=dpi, figsize=(1024 / dpi, 2048 / dpi))
mpl.rcParams.update({'font.size': 9})

plt.title('Относительная частота появления слов')

ax = plt.axes()
ax.xaxis.grid(True, zorder=1)

x = range(N)
y = common_vals / sum

plt.barh(x, y, height=0.2, color='red', label=sign, alpha=0.7, zorder=2)
plt.yticks(range(len(common_words)), common_words, rotation=1)
plt.legend(loc='upper right')
