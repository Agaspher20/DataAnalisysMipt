# В этом задании будет использоваться датасет digits из sklearn.datasets. Оставьте последние 25% объектов для
# контроля качества, разделив X и y на X_train, y_train и X_test, y_test.
# Целью задания будет реализовать самый простой метрический классификатор — метод ближайшего соседа, а также
# сравнить качество работы реализованного вами 1NN с RandomForestClassifier из sklearn на 1000 деревьях.
#%%
import numpy as np
from sklearn import datasets as dss
digits_dataset = dss.load_digits()
X = digits_dataset.data
y = digits_dataset.target
train_count = int((1.-0.25)*len(y))
X_train = X[:train_count]
y_train = y[:train_count]
X_test = X[train_count:]
y_test = y[train_count:]
target_names = digits_dataset.target_names

## Задание 1
# Реализуйте самостоятельно метод одного ближайшего соседа с евклидовой метрикой для задачи классификации.
# Можно не извлекать корень из суммы квадратов отклонений, т.к. корень — монотонное преобразование и не влияет
# на результат работы алгоритма.

# Никакой дополнительной работы с признаками в этом задании делать не нужно — мы еще успеем этим заняться в
# других курсах. Ваша реализация может быть устроена следующим образом: можно для каждого классифицируемого
# объекта составлять список пар (расстояние до точки из обучающей выборки, метка класса в этой точке), затем
# сортировать этот список (по умолчанию сортировка будет сначала по первому элементу пары, затем по второму),
# а затем брать первый элемент (с наименьшим расстоянием).

# Сортировка массива длиной N требует порядка N log N сравнений (строже говоря, она работает за O(N log N)).
# Подумайте, как можно легко улучшить получившееся время работы. Кроме простого способа найти ближайший объект
# всего за N сравнений, можно попробовать придумать, как разбить пространство признаков на части и сделать
# структуру данных, которая позволит быстро искать соседей каждой точки. За выбор метода поиска ближайших
# соседей в KNeighborsClassifier из sklearn отвечает параметр algorithm — если у вас уже есть некоторый
# бэкграунд в алгоритмах и структурах данных, вам может быть интересно познакомиться со структурами данных
# ball tree и kd tree.

# Доля ошибок, допускаемых 1NN на тестовой выборке, — ответ в задании 1.
#%%
train_pairs = zip(X_train, y_train)
test_pairs = zip(X_test, y_test)

def euclidean_distance(x1, x2):
    return sum(map(lambda (el1,el2): float(el1-el2)**2., zip(x1,x2)))

def predict(train_data, item, labels, k):
    item_vector, expected_label = item
    distances = map(lambda(train_vector, train_label):
                    (euclidean_distance(train_vector, item_vector), train_label),
                    train_data)
    neighbours = sorted(distances, key=lambda(dist, label): dist)[:k]
    counts = np.zeros(len(labels), dtype=int)
    for (_, label) in neighbours:
        idx = np.where(labels == label)[0][0]
        counts[idx] = counts[idx] + 1
    return (labels[np.argmax(counts)], expected_label)

predictions = map(lambda pair: predict(train_pairs, pair, target_names, 1), test_pairs)
#%%
false_count = np.sum(map(lambda (predicted, expected): predicted != expected, predictions))
answer1 = float(false_count)/float(len(test_pairs))
#%%
def write_answer_kfold(fileName, answer):
    with open("1nnVsRandomForest_" + fileName + ".txt", "w") as fout:
        fout.write(str(answer))
write_answer_kfold("1", answer1)
answer1


## Задание 2
# Теперь обучите на обучающей выборке RandomForestClassifier(n_estimators=1000) из sklearn. Сделайте прогнозы
# на тестовой выборке и оцените долю ошибок классификации на ней. Эта доля — ответ в задании 2. Обратите
# внимание на то, как соотносится качество работы случайного леса с качеством работы, пожалуй, одного из самых
# простых методов — 1NN. Такое различие — особенность данного датасета, но нужно всегда помнить, что такая
# ситуация тоже может иметь место, и не забывать про простые методы.
#%%
from sklearn.ensemble import RandomForestClassifier
forest_classifier = RandomForestClassifier(n_estimators=1000).fit(X_train, y_train)
#%%
forest_predictions = map(lambda (vector, expected_label):
                         (forest_classifier.predict(vector.reshape(1,-1))[0], expected_label),
                         test_pairs)
false_count = np.sum(map(lambda (predicted, expected): predicted != expected, forest_predictions))
answer2 = float(false_count)/float(len(test_pairs))
write_answer_kfold("2", answer2)
answer2
