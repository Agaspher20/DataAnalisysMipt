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

## Анализ основных недостатков метода главных компонент
# Рассмотренные выше задачи являются, безусловно, модельными, потому что данные для них были сгенерированы в
# соответствии с предположениями метода главных компонент. На практике эти предположения, естественно,
# выполняются далеко не всегда. Рассмотрим типичные ошибки PCA, которые следует иметь в виду перед тем, как его
# применять.
## Направления с максимальной дисперсией в данных неортогональны
# Рассмотрим случай выборки, которая сгенерирована из двух вытянутых нормальных распределений:
#%%
C1 = np.array([[10,0],[0,0.5]])
phi = np.pi/3
C2 = np.dot(C1, np.array([[np.cos(phi), np.sin(phi)],
                          [-np.sin(phi),np.cos(phi)]]))

data = np.vstack([np.random.multivariate_normal(mu, C1, size=50),
                  np.random.multivariate_normal(mu, C2, size=50)])
plt.scatter(data[:,0], data[:,1])
# построим истинные интересующие нас компоненты
plt.plot(data[:,0], np.zeros(data[:,0].size), color="g")
plt.plot(data[:,0], 3**0.5*data[:,0], color="g")
# обучим модель pca и построим главные компоненты
model = PCA(n_components=2)
model.fit(data)
plot_principal_components(data, model, scatter=False, legend=False)
c_patch = mpatches.Patch(color='c', label='Principal components')
plt.legend(handles=[g_patch, c_patch])
plt.draw()

# В чём проблема, почему pca здесь работает плохо? Ответ прост: интересующие нас компоненты в данных
# коррелированны между собой (или неортогональны, в зависимости от того, какой терминологией пользоваться).
# Для поиска подобных преобразований требуются более сложные методы, которые уже выходят за рамки метода
# главных компонент.
# Для интересующихся: то, что можно применить непосредственно к выходу метода главных компонент, для получения
# подобных неортогональных преобразований, называется методами ротации. Почитать о них можно в связи с другим
# методом уменьшения размерности, который называется Factor Analysis (FA), но ничего не мешает их применять и
# к главным компонентам.

## Интересное направление в данных не совпадает с направлением максимальной дисперсии
# Рассмотрим пример, когда дисперсии не отражают интересующих нас направлений в данных:
#%%
C = np.array([[0.5,0],[0,10]])
mu1 = np.array([-2,0])
mu2 = np.array([2,0])

data = np.vstack([np.random.multivariate_normal(mu1, C, size=50),
                  np.random.multivariate_normal(mu2, C, size=50)])
plt.scatter(data[:,0], data[:,1])
# обучим модель pca и построим главные компоненты
model = PCA(n_components=2)
model.fit(data)
plot_principal_components(data, model)
plt.draw()

# Очевидно, что в данном случае метод главных компонент будет считать вертикальную компоненту более значимой для
# описания набора данных, чем горизонтальную.
# Но, например, в случае, когда данные из левого и правого кластера относятся к разным классам, для их линейной
# разделимости вертикальная компонента является шумовой. Несмотря на это, её метод главных компонент никогда
# шумовой не признает, и есть вероятность, что отбор признаков с его помощью выкинет из ваших данных значимые
# для решаемой вами задачи компоненты просто потому, что вдоль них значения имеют низкую дисперсию.
# Справляться с такими ситуациями могут некоторые другие методы уменьшения размерности данных, например, метод
# независимых компонент (Independent Component Analysis, ICA).
