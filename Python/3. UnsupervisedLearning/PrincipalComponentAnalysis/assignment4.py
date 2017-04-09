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

## Интерпретация главных компонент с использованием данных
# Рассмотрим теперь величину, которую можно проинтерпретировать, как квадрат косинуса угла между объектом
# выборки и главной компонентой

## Задание 4. Анализ главных компонент при помощи вкладов в их дисперсию отдельных объектов
# 1. Загрузите датасет лиц Olivetti Faces и обучите на нём модель RandomizedPCA (используется при большом
#    количестве признаков и работает быстрее, чем обычный PCA). Получите проекции признаков на 10 первых главных
#    компонент.
# 2. Посчитайте для каждого объекта его относительный вклад в дисперсию каждой из 10 компонент, используя
#    формулу из предыдущего раздела (d = 10).
# 3. Для каждой компоненты найдите и визуализируйте лицо, которое вносит наибольший относительный вклад в неё.
#    Для визуализации используйте функцию
#       plt.imshow(image.reshape(image_shape))
# 4. Передайте в функцию write_answer_4 список номеров лиц с наибольшим относительным вкладом в дисперсию каждой
#    из компонент, список начинается с 0.
#%%
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import RandomizedPCA

def write_answer_4(list_pc):
    with open("..\..\..\Results\pca_answer4.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in list_pc]))

data = fetch_olivetti_faces(shuffle=True, random_state=0).data
image_shape = (64, 64)
#%%
components_count = 10
model = RandomizedPCA(components_count)
model.fit(data)
faces_transformed = model.transform(data)
#%%
def center_features(matrix):
    matrix_t = matrix.transpose()
    means = [np.mean(col) for col in matrix_t]
    matrix_t_centered = [[item-col[1] for item in col[0]] for col in zip(matrix_t, means)]
    return np.array(matrix_t_centered).transpose()
#%%
data_centered = center_features(faces_transformed)
#%%
data_contributions = []
for face in data_centered:
    squares = map(lambda f: f**2., face)
    projections_sum = sum(squares)
    data_contributions.append([proj_sq/projections_sum for proj_sq in squares])
#%%
component_faces = []
for contribution_column in np.array(data_contributions).transpose():
    max_face = max(enumerate(contribution_column), key=lambda f: f[1])
    component_faces.append(max_face[0])
#%%
face = data[component_faces[0]]
plt.imshow(face.reshape(image_shape))
#%%
write_answer_4(component_faces)
