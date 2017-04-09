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

## Пример
# Рассмотрим набор данных, который сэмплирован из многомерного нормального распределения с матрицей ковариации 
# C
#%%
from sklearn.decomposition import PCA

mu = np.zeros(2)
C = np.array([[3,1],[1,2]])

data = np.random.multivariate_normal(mu, C, size=50)
plt.scatter(data[:,0], data[:,1])
plt.show()
# Путём диагонализации истинной матрицы ковариаций C, мы можем найти преобразование исходного набора данных,
# компоненты которого наилучшим образом будут описывать дисперсию, с учётом их ортогональности друг другу:
#%%
v, W_true = np.linalg.eig(C)

plt.scatter(data[:,0], data[:,1])
# построим истинные компоненты, вдоль которых максимальна дисперсия данных
plt.plot(data[:,0], (W_true[0,0]/W_true[0,1])*data[:,0], color="g")
plt.plot(data[:,0], (W_true[1,0]/W_true[1,1])*data[:,0], color="g")
g_patch = mpatches.Patch(color="g", label="True components")
plt.legend(handles=[g_patch])
plt.axis("equal")
limits = [np.minimum(np.amin(data[:,0]), np.amin(data[:,1])),
          np.maximum(np.amax(data[:,0]), np.amax(data[:,1]))]
plt.xlim(limits[0],limits[1])
plt.ylim(limits[0],limits[1])
plt.draw()
# А теперь сравним эти направления с направлениями, которые выбирает метод главных компонент:
#%%
def plot_principal_components(data, model, scatter=True, legend=True):
    W_pca = model.components_
    if scatter:
        plt.scatter(data[:,0], data[:,1], alpha=0.3)
    plt.plot(data[:,0], -(W_pca[0,0]/W_pca[0,1])*data[:,0], color="c")
    plt.plot(data[:,0], -(W_pca[1,0]/W_pca[1,1])*data[:,0], color="c")
    if legend:
        c_patch = mpatches.Patch(color="c", label="Principal components")
        plt.legend(handles=[c_patch], loc="lower right")
    # сделаем графики красивыми:
    plt.axis("equal")
    limits = [np.minimum(np.amin(data[:,0]), np.amin(data[:,1]))-0.5,
              np.maximum(np.amax(data[:,0]), np.amax(data[:,1]))+0.5]
    plt.xlim(limits[0],limits[1])
    plt.ylim(limits[0],limits[1])
    plt.draw()
#%%
model = PCA(n_components=2)
model.fit(data)

# построим истинные компоненты, вдоль которых максимальна дисперсия данных
plt.plot(data[:,0], (W_true[0,0]/W_true[0,1])*data[:,0], color="g")
plt.plot(data[:,0], (W_true[1,0]/W_true[1,1])*data[:,0], color="g")
# построим компоненты, полученные с использованием метода PCA:
plot_principal_components(data, model, legend=False)
c_patch = mpatches.Patch(color="c", label="Principal components")
plt.legend(handles=[g_patch, c_patch])
plt.draw()
# Видно, что уже при небольшом количестве данных они отличаются незначительно. Увеличим размер выборки:
#%%
data_large = np.random.multivariate_normal(mu, C, size=5000)

model = PCA(n_components=2)
model.fit(data_large)
# построим истинные компоненты, вдоль которых максимальна дисперсия данных
plt.plot(data_large[:,0], (W_true[0,0]/W_true[0,1])*data_large[:,0], color="g")
plt.plot(data_large[:,0], (W_true[1,0]/W_true[1,1])*data_large[:,0], color="g")
# построим компоненты, полученные с использованием метода PCA:
plot_principal_components(data_large, model, legend=False)
c_patch = mpatches.Patch(color="c", label="Principal components")
plt.legend(handles=[g_patch, c_patch])
plt.draw()
# В этом случае главные компоненты значительно точнее приближают истинные направления данных, вдоль которых
# наблюдается наибольшая дисперсия.
