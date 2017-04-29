## Programming Assignment:

## Готовим LDA по рецептам
# Как вы уже знаете, в тематическом моделировании делается предположение о том, что для определения тематики
# порядок слов в документе не важен; об этом гласит гипотеза «мешка слов». Сегодня мы будем работать с несколько
# нестандартной для тематического моделирования коллекцией, которую можно назвать «мешком ингредиентов», потому
# что она состоит из рецептов блюд разных кухонь. Тематические модели ищут слова, которые часто вместе
# встречаются в документах, и составляют из них темы. Мы попробуем применить эту идею к рецептам и найти
# кулинарные «темы». Эта коллекция хороша тем, что не требует предобработки. Кроме того, эта задача достаточно
# наглядно иллюстрирует принцип работы тематических моделей.

# Для выполнения заданий, помимо часто используемых в курсе библиотек, потребуются модули json и gensim. Первый
# входит в дистрибутив Anaconda, второй можно поставить командой
#   pip install gensim
# Построение модели занимает некоторое время. На ноутбуке с процессором Intel Core i7 и тактовой частотой
# 2400 МГц на построение одной модели уходит менее 10 минут.

## Загрузка данных
# Коллекция дана в json-формате: для каждого рецепта известны его id, кухня (cuisine) и список ингредиентов, в
# него входящих. Загрузить данные можно с помощью модуля json (он входит в дистрибутив Anaconda):

#%%
import json
#%%
with open("recipes.json") as f:
    recipes = json.load(f)
#%%
print recipes[0]

## Составление корпуса
#%%
from gensim import corpora, models
import numpy as np

# Наша коллекция небольшая, и целиком помещается в оперативную память. Gensim может работать с такими данными и
# не требует их сохранения на диск в специальном формате. Для этого коллекция должна быть представлена в виде
# списка списков, каждый внутренний список соответствует отдельному документу и состоит из его слов. Пример
# коллекции из двух документов: 
#   [["hello", "world"], ["programming", "in", "python"]]
# Преобразуем наши данные в такой формат, а затем создадим объекты corpus и dictionary, с которыми будет
# работать модель.
#%%
texts = [recipe["ingredients"] for recipe in recipes]
dictionary = corpora.Dictionary(texts)   # составляем словарь
corpus = [dictionary.doc2bow(text) for text in texts]  # составляем корпус документов
#%%
print texts[0]
print corpus[0]

## Обучение модели
# Вам может понадобиться документация LDA в gensim.
# Задание 1. Обучите модель LDA с 40 темами, установив количество проходов по коллекции 5 и оставив остальные
# параметры по умолчанию. 
#%%
np.random.seed(76543)
# здесь код для построения модели:
model = models.ldamodel.LdaModel(corpus, num_topics=40, passes=5)
# Затем вызовите метод модели show_topics, указав количество тем 40 и количество токенов 10, и сохраните
# результат (топы ингредиентов в темах) в отдельную переменную. Если при вызове метода show_topics указать
# параметр formatted=True, то топы ингредиентов будет удобно выводить на печать, если formatted=False, будет
# удобно работать со списком программно. Выведите топы на печать, рассмотрите темы, а затем ответьте на вопрос:
#%%
top_topics = model.show_topics(num_topics=40, num_words=10, formatted=False)

# Сколько раз ингредиенты "salt", "sugar", "water", "mushrooms", "chicken", "eggs" встретились среди топов-10
# всех 40 тем? При ответе не нужно учитывать составные ингредиенты, например, "hot water".
#%%
salt_count = 0
sugar_count = 0
water_count = 0
mushrooms_count = 0
chicken_count = 0
eggs_count = 0
for theme in top_topics:
    print "{}: ".format(theme[0])
    for token in theme[1]:
        word = dictionary[int(token[0])]
        print "\t", word, "({}) score: {}".format(token[0], token[1])
        if(word == 'salt'): salt_count = salt_count + 1
        if(word == 'sugar'): sugar_count = sugar_count + 1
        if(word == 'water'): water_count = water_count + 1
        if(word == 'mushrooms'): mushrooms_count = mushrooms_count + 1
        if(word == 'chicken'): chicken_count = chicken_count + 1
        if(word == 'eggs'): eggs_count = eggs_count + 1

# Передайте 6 чисел в функцию save_answers1 и загрузите сгенерированный файл в форму.
# У gensim нет возможности фиксировать случайное приближение через параметры метода, но библиотека использует
# numpy для инициализации матриц. Поэтому, по утверждению автора библиотеки, фиксировать случайное приближение
# нужно командой, которая написана в следующей ячейке. Перед строкой кода с построением модели обязательно
# вставляйте указанную строку фиксации random.seed.
#%%
def save_answers1(c_salt, c_sugar, c_water, c_mushrooms, c_chicken, c_eggs):
    with open("cooking_LDA_pa_task1.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [c_salt, c_sugar, c_water, c_mushrooms, c_chicken, c_eggs]]))
#%%
print salt_count, sugar_count, water_count, mushrooms_count, chicken_count, eggs_count
save_answers1(salt_count, sugar_count, water_count, mushrooms_count, chicken_count, eggs_count)

## Фильтрация словаря
# В топах тем гораздо чаще встречаются первые три рассмотренных ингредиента, чем последние три. При этом
# наличие в рецепте курицы, яиц и грибов яснее дает понять, что мы будем готовить, чем наличие соли, сахара и
# воды. Таким образом, даже в рецептах есть слова, часто встречающиеся в текстах и не несущие смысловой
# нагрузки, и поэтому их не желательно видеть в темах. Наиболее простой прием борьбы с такими фоновыми
# элементами — фильтрация словаря по частоте. Обычно словарь фильтруют с двух сторон: убирают очень редкие слова
# (в целях экономии памяти) и очень частые слова (в целях повышения интерпретируемости тем). Мы уберем только
# частые слова.
#%%
import copy
dictionary2 = copy.deepcopy(dictionary)

# Задание 2. У объекта dictionary2 есть переменная dfs — это словарь, ключами которого являются id токена, а
# элементами — число раз, сколько слово встретилось во всей коллекции. Сохраните в отдельный список ингредиенты,
# которые встретились в коллекции больше 4000 раз. Вызовите метод словаря filter_tokens, подав в качестве
# первого аргумента полученный список популярных ингредиентов. Вычислите две величины: dict_size_before и
# dict_size_after — размер словаря до и после фильтрации.
#%%
frequent_tokens_keys = []
frequent_tokens = []
for k in dictionary2.dfs:
    frequency = dictionary2.dfs[k]
    if(frequency > 4000):
        frequent_tokens_keys.append(k)
        frequent_tokens.append(dictionary2[k])

dict_size_before = len(dictionary2)
dictionary2.filter_tokens(frequent_tokens_keys)
dict_size_after = len(dictionary2)
print dict_size_before, dict_size_after, len(frequent_tokens)
print frequent_tokens

# Затем, используя новый словарь, создайте новый корпус документов, corpus2, по аналогии с тем, как это сделано
# в начале ноутбука.
#%%
corpus2 = [dictionary2.doc2bow(text) for text in texts]  # составляем корпус документов

# Вычислите две величины: corpus_size_before и corpus_size_after — суммарное количество
# ингредиентов в корпусе (для каждого документа вычислите число различных ингредиентов в нем и просуммируйте по
# всем документам) до и после фильтрации.
#%%
def calculate_corpus_size(corp):
    corpus_size = 0
    for doc in corp:
        corpus_size = corpus_size + len(doc)
    return corpus_size
corpus_size_before = calculate_corpus_size(corpus)
corpus_size_after = calculate_corpus_size(corpus2)
print corpus_size_before, corpus_size_after

# Передайте величины dict_size_before, dict_size_after, corpus_size_before, corpus_size_after в функцию
# save_answers2 и загрузите сгенерированный файл в форму.
#%%
def save_answers2(dict_size_before, dict_size_after, corpus_size_before, corpus_size_after):
    with open("cooking_LDA_pa_task2.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [dict_size_before, dict_size_after, corpus_size_before, corpus_size_after]]))
#%%
print dict_size_before, dict_size_after, corpus_size_before, corpus_size_after
save_answers2(dict_size_before, dict_size_after, corpus_size_before, corpus_size_after)

## Сравнение когерентностей
# Задание 3. Постройте еще одну модель по корпусу corpus2 и словарю dictionary2, остальные параметры оставьте
# такими же, как при первом построении модели. Сохраните новую модель в другую переменную (не перезаписывайте
# предыдущую модель). Не забудьте про фиксирование seed!
#%%
np.random.seed(76543)
model2 = models.ldamodel.LdaModel(corpus2, num_topics=40, passes=5)

# Затем воспользуйтесь методом top_topics модели, чтобы вычислить ее когерентность. Передайте в качестве
# аргумента соответствующий модели корпус. Метод вернет список кортежей (топ токенов, когерентность),
# отсортированных по убыванию последней. Вычислите среднюю по всем темам когерентность для каждой из двух
# моделей и передайте в функцию save_answers3.
#%%
top_coherences = model.top_topics(corpus)
top_coherences2 = model2.top_topics(corpus2)
#%%
avg_coherence = np.average(map(lambda topic: topic[1], top_coherences))
avg_coherence2 = np.average(map(lambda topic: topic[1], top_coherences2))
#%%
def save_answers3(coherence, coherence2):
    with open("cooking_LDA_pa_task3.txt", "w") as fout:
        fout.write(" ".join(["%3f"%el for el in [coherence, coherence2]]))
#%%
print avg_coherence, avg_coherence2
save_answers3(avg_coherence, avg_coherence2)

# Считается, что когерентность хорошо соотносится с человеческими оценками интерпретируемости тем. Поэтому на
# больших текстовых коллекциях когерентность обычно повышается, если убрать фоновую лексику. Однако в нашем
# случае этого не произошло.

## Изучение влияния гиперпараметра alpha
# В этом разделе мы будем работать со второй моделью, то есть той, которая построена по сокращенному корпусу.

# Пока что мы посмотрели только на матрицу темы-слова, теперь давайте посмотрим на матрицу темы-документы.
# Выведите темы для нулевого (или любого другого) документа из корпуса, воспользовавшись методом
# get_document_topics второй модели:
#%%
document_topics2 = model2.get_document_topics(corpus2[0])
# Также выведите содержимое переменной .alpha второй модели:
#%%
print "Themes:", document_topics2
print "Alpha:", model2.alpha[0]
# У вас должно получиться, что документ характеризуется небольшим числом тем. Попробуем поменять гиперпараметр
# alpha, задающий априорное распределение Дирихле для распределений тем в документах.

# Задание 4. Обучите третью модель: используйте сокращенный корпус (corpus2 и dictionary2) и установите параметр
# alpha=1, passes=5. Не забудьте про фиксацию seed!
#%%
np.random.seed(76543)
model3 = models.ldamodel.LdaModel(corpus2, num_topics=40, passes=5, alpha=1.)

# Выведите темы новой модели для нулевого документа; должно получиться, что распределение над множеством тем
# практически равномерное. Чтобы убедиться в том, что во второй модели документы описываются гораздо более
# разреженными распределениями, чем в третьей, посчитайте суммарное количество элементов, превосходящих 0.01,
# в матрицах темы-документы обеих моделей. Другими словами, запросите темы модели для каждого документа с
# параметром minimum_probability=0.01 и просуммируйте число элементов в получаемых массивах.
#%%
document_topics2_001 = model2.get_document_topics(corpus2[0], minimum_probability=0.01)
document_topics3 = model3.get_document_topics(corpus2[0], minimum_probability=0.01)
#%%
print "Themes 2:", document_topics2_001
print "Themes 3:", document_topics3
print "Alpha 3:", model3.alpha[0]
# Передайте две суммы (сначала для модели с alpha по умолчанию, затем для модели в alpha=1)
# в функцию save_answers4.
#%%
def save_answers4(count_model2, count_model3):
    with open("cooking_LDA_pa_task4.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [count_model2, count_model3]]))
#%%
def calculate_document_topics_sum(mdl, corp):
    corp_sum = 0.
    for i in range(0, len(corp)):
        doc_topics = mdl.get_document_topics(corp[i], minimum_probability=0.01)
        corp_sum = corp_sum + len(doc_topics)
    return corp_sum
#%%
document_topics_sum2 = calculate_document_topics_sum(model2, corpus2)
document_topics_sum3 = calculate_document_topics_sum(model3, corpus2)
print document_topics_sum2, document_topics_sum3
save_answers4(document_topics_sum2, document_topics_sum3)

# Таким образом, гиперпараметр alpha влияет на разреженность распределений тем в документах. Аналогично
# гиперпараметр eta влияет на разреженность распределений слов в темах.

## LDA как способ понижения размерности
# Иногда, распределения над темами, найденные с помощью LDA, добавляют в матрицу объекты-признаки как
# дополнительные, семантические, признаки, и это может улучшить качество решения задачи. Для простоты давайте
# просто обучим классификатор рецептов на кухни на признаках, полученных из LDA, и измерим точность (accuracy).

# Задание 5. Используйте модель, построенную по сокращенной выборке с alpha по умолчанию (вторую модель).
# Составьте матрицу Θ=p(t|d) вероятностей тем в документах; вы можете использовать тот же метод
# get_document_topics, а также вектор правильных ответов y (в том же порядке, в котором рецепты идут в
# переменной recipes). Создайте объект RandomForestClassifier со 100 деревьями, с помощью функции
# cross_val_score вычислите среднюю accuracy по трем фолдам (перемешивать данные не нужно) и передайте в
# функцию save_answers5.

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

#%%
def build_probability_matrix(corpus, model):
    overall_max_topic = 0
    documents_topics = []
    for i,doc in enumerate(corpus):
        document_topics = model.get_document_topics(doc, minimum_probability=0)
        documents_topics.append(document_topics)
        max_topic = max(document_topics, key=lambda topic: topic[0])[0]
        if(max_topic > overall_max_topic):
            overall_max_topic = max_topic
    
    probability_matrix = np.zeros((len(documents_topics), overall_max_topic + 1))
    for i,document_topics in enumerate(documents_topics):
        for topic in document_topics:
            probability_matrix[i][topic[0]] = topic[1]
    return probability_matrix
#%%
import itertools as it
true_responses = map(lambda recipe: recipe['cuisine'], recipes)
#%%
forest_classifier = RandomForestClassifier(n_estimators=100)
X = build_probability_matrix(corpus2, model2)
y = np.array(true_responses)
print X.shape, y.shape

scores = cross_val_score(forest_classifier, X, y, cv=3)

#%%
forest_classifier = RandomForestClassifier(n_estimators=100)
X = build_probability_matrix(corpus2, model2)
y = np.array(true_responses)
print X.shape, y.shape

scores = cross_val_score(forest_classifier, X, y, cv=3)
        
#%%
def save_answers5(accuracy):
     with open("cooking_LDA_pa_task5.txt", "w") as fout:
        fout.write(str(accuracy))
#%%
mean_score = scores.mean()
print mean_score
save_answers5(mean_score)

# Для такого большого количества классов это неплохая точность. Вы можете попроовать обучать RandomForest на
# исходной матрице частот слов, имеющей значительно большую размерность, и увидеть, что accuracy увеличивается
# на 10–15%. Таким образом, LDA собрал не всю, но достаточно большую часть информации из выборки, в матрице
# низкого ранга.

## LDA — вероятностная модель
# Матричное разложение, использующееся в LDA, интерпретируется как следующий процесс генерации документов.
# Для документа d длины nd:
# 1. Из априорного распределения Дирихле с параметром alpha сгенерировать распределение над множеством тем:
#    θd ∼ Dirichlet(α)
# 2. Для каждого слова w=1,…,nd:
#       A. Сгенерировать тему из дискретного распределения t ∼ θd
#       B. Сгенерировать слово из дискретного распределения w ∼ ϕt.

# Подробнее об этом в Википедии.
# В контексте нашей задачи получается, что, используя данный генеративный процесс, можно создавать новые
# рецепты. Вы можете передать в функцию модель и число ингредиентов и сгенерировать рецепт :)
#%%
def generate_recipe(model, num_ingredients):
    theta = np.random.dirichlet(model.alpha)
    for i in range(num_ingredients):
        t = np.random.choice(np.arange(model.num_topics), p=theta)
        topic = model.show_topic(t, topn=model.num_terms)
        topic_distr = [x[1] for x in topic]
        terms = [x[0] for x in topic]
        w = np.random.choice(terms, p=topic_distr)
        print w

#%%
## Интерпретация построенной модели
# Вы можете рассмотреть топы ингредиентов каждой темы. Большиснтво тем сами по себе похожи на рецепты; в
# некоторых собираются продукты одного вида, например, свежие фрукты или разные виды сыра.
# Попробуем эмпирически соотнести наши темы с национальными кухнями (cuisine). Построим матрицу A размера темы
# x кухни, ее элементы atc — суммы p(t|d) по всем документам d, которые отнесены к кухне c. Нормируем матрицу
# на частоты рецептов по разным кухням, чтобы избежать дисбаланса между кухнями. Следующая функция получает на
# вход объект модели, объект корпуса и исходные данные и возвращает нормированную матрицу A.
# Ее удобно визуализировать с помощью seaborn.
#%%
import pandas
import seaborn
from matplotlib import pyplot as plt
%matplotlib inline
#%%
def compute_topic_cuisine_matrix(model, corpus, recipes):
    # составляем вектор целевых признаков
    targets = list(set([recipe["cuisine"] for recipe in recipes]))
    # составляем матрицу
    tc_matrix = pandas.DataFrame(data=np.zeros((model.num_topics, len(targets))), columns=targets)
    for recipe, bow in zip(recipes, corpus):
        recipe_topic = model.get_document_topics(bow)
        for t, prob in recipe_topic:
            tc_matrix[recipe["cuisine"]][t] += prob
    # нормируем матрицу
    target_sums = pandas.DataFrame(data=np.zeros((1, len(targets))), columns=targets)
    for recipe in recipes:
        target_sums[recipe["cuisine"]] += 1
    return pandas.DataFrame(tc_matrix.values/target_sums.values, columns=tc_matrix.columns)
#%%
def plot_matrix(tc_matrix):
    plt.figure(figsize=(10, 10))
    seaborn.heatmap(tc_matrix, square=True)
#%%
# Визуализируйте матрицу

# Чем темнее квадрат в матрице, тем больше связь этой темы с данной кухней. Мы видим, что у нас есть темы,
# которые связаны с несколькими кухнями. Такие темы показывают набор ингредиентов, которые популярны в кухнях
# нескольких народов, то есть указывают на схожесть кухонь этих народов. Некоторые темы распределены по всем
# кухням равномерно, они показывают наборы продуктов, которые часто используются в кулинарии всех стран.

# Жаль, что в датасете нет названий рецептов, иначе темы было бы проще интерпретировать...

## Заключение
# В этом задании вы построили несколько моделей LDA, посмотрели, на что влияют гиперпараметры модели и как
# можно использовать построенную модель.
