# Astrology-ml
Проект был написан Нагорной Анной Дмитриевной М8О-206Б-21, Деревянко Екатериной Андреевной М8О-206Б-21 и Бондарь Миланой Олеговной М8О-206Б-21 в качестве курсового проекта по предмету ООП. Продукт - сайт на основе Django, который с помощью обученной модели генерирует гороскоп на сегодня для любого знака зодиака.

## Структура проекта
- astrology-site
- - [astrology-site](#astrology-site)
- - [main](#main)
- [model](#model)

### astrology-site
[astrology-site](/astrology_site/astrology_site/) - Основа для сайта на Django
- [settings](/astrology_site/astrology_site/settings.py) - Настройки для Django
- [urls](/astrology_site/astrology_site/urls.py) - Основные пути на сайте

### main
[main](/astrology_site/main/) - Приложение с генерацией гороскопов
- [templates](/astrology_site/main/templates) - Фронт
- [urls](/astrology_site/main/urls.py) - url для приложения
- [views](/astrology_site/main/views.py) - Представления для главной страницы
- [adapter](/astrology_site/main/adapter.py) - адаптер для класса подключения модели к сайту

### model
[main](/model) - Обучение модели
- [For_train_and_gen.ipynb](model/For_train_and_gen.ipynb) - Блокнот с обучением модели
- [horoscopes.csv](model/horoscopes.csv) - Первичный датасет. Распарсенная страница сайта с гороскопами в формате csv
- [negative.txt](model/negative.txt) - Отфильтрованные негативные предсказания для более точного обучения модели
- [positive.txt](model/positive.txt) - То же самое, только позитивные предсказания
- [parsing_dataset.py](model/parsing_dataset.py) - Подготовка датасета для обучения модели. На вход берет horoscopes.csv и отдает результат в text.csv
- [text.csv](model/text.csv) - Готовый датасет для обучения

### Запуск
#### С помощью терминала (Linux)
git clone https://github.com/FoxLand-b4n/Astrology-ml.git
pip install -r ./Astrology-ml/requirements.txt
python Astrology-ml/astrology_site/manage.py runserver

#### С помощью Dockerfile
`docker build -t astro .`

`docker run astro`

Или можно скачать образ с dockerhub

`docker pull da44it/astro`

`docker run astro`