""" Рекомендательные системы """
# Описание задачи
# Небольшой интернет-магазин попросил вас добавить ранжирование товаров в блок "Смотрели ранее"
# - в нем теперь надо показывать не последние просмотренные пользователем товары, а те товары из
# просмотренных, которые он наиболее вероятно купит. Качество вашего решения будет оцениваться по
# количеству покупок в сравнении с прошлым решением в ходе А/В теста, т.к. по доходу от продаж
# статзначимость будет достигаться дольше из-за разброса цен.
# Таким образом, ничего заранее не зная про корреляцию оффлайновых и онлайновых метрик качества,
# в начале проекта вы можете лишь постараться оптимизировать recall@k и precision@k.

# Это задание посвящено построению простых бейзлайнов для этой задачи:
# ранжирование просмотренных товаров по частоте просмотров и по частоте покупок.
# Эти бейзлайны, с одной стороны, могут помочь вам грубо оценить возможный эффект от ранжирования
# товаров в блоке - например, чтобы вписать какие-то числа в коммерческое предложение заказчику,
# а с другой стороны, могут оказаться самым хорошим вариантом, если данных очень мало
# (недостаточно для обучения даже простых моделей).

# Входные данные
# Вам дается две выборки с пользовательскими сессиями - id-шниками просмотренных и id-шниками
# купленных товаров.
# Одна выборка будет использоваться для обучения (оценки популярностей товаров), а другая
# - для теста.
# В файлах записаны сессии по одной в каждой строке.
# Формат сессии: id просмотренных товаров через , затем идёт ; после чего следуют id купленных
# товаров (если такие имеются), разделённые запятой. Например, 1,2,3,4; или 1,2,3,4;5,6.
# Гарантируется, что среди id купленных товаров все различные.
#%%
import sys
import pandas as pd
import numpy as np
sys.path.append("DataAnalisysMipt\\Python\\5. DataAnalysisApplications")
#%%
data_train = pd.read_csv(
    "DataAnalisysMipt\\Data\\coursera_sessions_train.txt",
    ";",
    header=0,
    names=["viewed", "bought"])
data_train.head()
#%%
data_test = pd.read_csv(
    "DataAnalisysMipt\\Data\\coursera_sessions_test.txt",
    ";",
    header=0,
    names=["viewed", "bought"])
data_test.head()

# Важно:
# Сессии, в которых пользователь ничего не купил, исключаем из оценки качества.
# Если товар не встречался в обучающей выборке, его популярность равна 0.
# Рекомендуем разные товары. И их число должно быть не больше, чем количество различных
# просмотренных пользователем товаров.
# Рекомендаций всегда не больше, чем минимум из двух чисел: количество просмотренных пользователем
# товаров и k в recall@k / precision@k.

# Задание
# 1. На обучении постройте частоты появления id в просмотренных и в купленных
#    (id может несколько раз появляться в просмотренных, все появления надо учитывать)
#%%
def parse_session_column(column):
    """ parses string from session column """
    return [int(val) for val in column.split(",")]
def parse_session(views, buys):
    """ parses session string """
    return (parse_session_column(views),
            parse_session_column(buys) if isinstance(buys, str) else [])
def update_frequencies_count(key, frequencies):
    """ increments dictionary value if key is present in dictionary or puts this key """
    frequencies[key] = frequencies[key] + 1 if key in frequencies else 1
def build_data_frequencies(data):
    """ counts product frequencies in views and buys """
    view_freqs = {}
    buy_freqs = {}
    for views_col,buys_col in data.as_matrix():
        views,buys = parse_session(views_col, buys_col)
        for view_key in views:
            update_frequencies_count(view_key, view_freqs)
        for buy_key in buys:
            update_frequencies_count(buy_key, buy_freqs)
    return view_freqs,buy_freqs

#%%
view_frequencies, buy_frequencies = build_data_frequencies(data_train)
    
# 2. Реализуйте два алгоритма рекомендаций:
#    сортировка просмотренных id по популярности (частота появления в просмотренных),
#    сортировка просмотренных id по покупаемости (частота появления в покупках).
#%%
view_buy_frequencies = {}
for k, v in view_frequencies.items():
    view_buy_frequencies[k] = buy_frequencies[k] if k in buy_frequencies else 0

sorted_view_freqs = [(k, v)
                     for k, v
                     in sorted(view_frequencies.items(), key=lambda kvp: kvp[1], reverse=True)]
sorted_buy_freqs = [(k, v)
                    for k, v
                    in sorted(view_buy_frequencies.items(), key=lambda kvp: kvp[1], reverse=True)]
# 3. Для данных алгоритмов выпишите через пробел AverageRecall@1, AveragePrecision@1,
#    AverageRecall@5, AveragePrecision@5 на обучающей и тестовых выборках, округляя до 2 знака
#    после запятой. Это будут ваши ответы в этом задании. Посмотрите, как они соотносятся друг с
#    другом.
#    Где качество получилось выше? Значимо ли это различие? Обратите внимание на различие качества
#    на обучающей и тестовой выборке в случае рекомендаций по частотам покупки.
# Если частота одинаковая, то сортировать нужно по возрастанию момента просмотра (чем раньше
# появился в просмотренных, тем больше приоритет)

# Дополнительные вопросы
# 1. Обратите внимание, что при сортировке по покупаемости возникает много товаров с одинаковым
#    рангом - это означает, что значение метрик будет зависеть от того, как мы будем сортировать
#    товары с одинаковым рангом. Попробуйте убедиться, что при изменении сортировки таких товаров
#    recall@k меняется. Подумайте, как оценить минимальное и максимальное значение recall@k в
#    зависимости от правила сортировки.
# 2. Мы обучаемся и тестируемся на полных сессиях (в которых есть все просмотренные за сессию
#    товары).
#    Подумайте, почему полученная нами оценка качества рекомендаций в этом случае несколько
#    завышена.
