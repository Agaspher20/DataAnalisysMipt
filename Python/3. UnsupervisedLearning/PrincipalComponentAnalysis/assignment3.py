## Метод главных компонент
# В данном задании вам будет предложено ознакомиться с подходом, который переоткрывался в самых разных областях,
# имеет множество разных интерпретаций, а также несколько интересных обобщений: методом главных компонент
# (principal component analysis).

## Programming assignment
# Задание разбито на две части: 
# работа с модельными данными,
# работа с реальными данными.
# В конце каждого пункта от вас требуется получить ответ и загрузить в соответствующую форму в виде набора
# текстовых файлов.
#%%
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
matplotlib.style.use("ggplot")
%matplotlib inline

from sklearn.decomposition import PCA

## Интерпретация главных компонент
# В качестве главных компонент мы получаем линейные комбинации исходных призанков, поэтому резонно возникает
# вопрос об их интерпретации.
# Для этого существует несколько подходов, мы рассмотрим два:
# 1. рассчитать взаимосвязи главных компонент с исходными признаками
# 2. рассчитать вклады каждого конкретного наблюдения в главные компоненты
# Первый способ подходит в том случае, когда все объекты из набора данных не несут для нас никакой семантической
# информации, которая уже не запечатлена в наборе признаков.
# Второй способ подходит для случая, когда данные имеют более сложную структуру. Например, лица для человека
# несут больший семантический смысл, чем вектор значений пикселей, которые анализирует PCA.
# Рассмотрим подробнее способ 1: он заключается в подсчёте коэффициентов корреляций между исходными признаками
# и набором главных компонент.
# Так как метод главных компонент является линейным, то предлагается для анализа использовать корреляцию Пирсона.
# Корреляция Пирсона является мерой линейной зависимости. Она равна 0 в случае, когда величины независимы, и
# ±1, если они линейно зависимы. Исходя из степени корреляции новой компоненты с исходными признаками, можно
# строить её семантическую интерпретацию, т.к. смысл исходных признаков мы знаем.

## Задание 3. Анализ главных компонент при помощи корреляций с исходными признаками.
# 1. Обучите метод главных компонент на датасете iris, получите преобразованные данные.
# 2. Посчитайте корреляции исходных признаков с их проекциями на первые две главные компоненты.
# 3. Для каждого признака найдите компоненту (из двух построенных), с которой он коррелирует больше всего.
# 4. На основании п.3 сгруппируйте признаки по компонентам. Составьте два списка: список номеров признаков,
#    которые сильнее коррелируют с первой компонентой, и такой же список для второй. Нумерацию начинать с
#    единицы. Передайте оба списка функции write_answer_3.
# Набор данных состоит из 4 признаков, посчитанных для 150 ирисов. Каждый из них принадлежит одному из трёх видов.
# Визуализацию проекции данного датасета на две компоненты, которые описывают наибольшую дисперсию данных, можно
# получить при помощи функции
#   plot_iris(transformed_data, target, target_names)
# на вход которой требуется передать данные, преобразованные при помощи PCA, а также информацию о классах. Цвет
# точек отвечает одному из трёх видов ириса.
# Для того чтобы получить имена исходных признаков, используйте следующий список:
#   iris.feature_names
# При подсчёте корреляций не забудьте центрировать признаки и проекции на главные компоненты (вычитать из них
# среднее).
#%%
from sklearn import datasets

def plot_iris(transformed_data, target, target_names):
    plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
        plt.scatter(transformed_data[target == i, 0],
                    transformed_data[target == i, 1], c=c, label=target_name)
    plt.legend()
    plt.show()
    
def write_answer_3(list_pc1, list_pc2):
    with open("..\..\..\Results\pca_answer3.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in list_pc1]))
        fout.write(" ")
        fout.write(" ".join([str(num) for num in list_pc2]))

# загрузим датасет iris
iris = datasets.load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names

#%%
model = PCA(n_components=2)
model.fit(data)
iris_components = model.transform(data)
data_t = data.transpose()
components_t = iris_components.transpose()
data_means = [np.mean(col) for col in data_t]
components_means = [np.mean(col) for col in components_t]
data_centered = [[item - col[1] for item in col[0]] for col in zip(data_t, data_means)]
components_centered = [[item - col[1] for item in col[0]] for col in zip(components_t, components_means)]

first_component_features = []
second_component_features = []
max_correlations = []
for i,feature in enumerate(data_centered):
    correlations = [(j,np.correlate(feature, component)) for j,component in enumerate(components_centered)]
    max_corr = max(correlations, key=lambda c: c[1])
    if max_corr[0] == 0:
        first_component_features.append(i+1)
    else:
        second_component_features.append(i+1)
    max_correlations.append((i,max_corr[0],max_corr[1][0]))
    
plot_iris(np.array(components_centered).transpose(), target, target_names)
write_answer_3(first_component_features, second_component_features)
