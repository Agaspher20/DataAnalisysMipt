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

## Задание 1. Автоматическое уменьшение размерности данных при помощи логарифма правдоподобия
# Рассмотрим набор данных размерности D, чья реальная размерность значительно меньше наблюдаемой (назовём её
# d). От вас требуется:
# 1. Для каждого значения d^ в интервале [1,D] построить модель PCA с d^ главными компонентами.
# 2. Оценить средний логарифм правдоподобия данных для каждой модели на генеральной совокупности, используя
#    метод кросс-валидации с 3 фолдами (итоговая оценка значения логарифма правдоподобия усредняется по всем
#    фолдам).
# 3. Найти модель, для которой он максимален, и внести в файл ответа число компонент в данной модели, т.е.
#    значение d^opt.
#%%
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score as cv_score

def plot_scores(d_scores):
    n_components = np.arange(1,d_scores.size+1)
    plt.plot(n_components, d_scores, "b", label="PCA scores")
    plt.xlim(n_components[0], n_components[-1])
    plt.xlabel("n components")
    plt.ylabel("cv scores")
    plt.legend(loc="lower right")
    plt.show()
    
def write_answer_1(optimal_d):
    with open("..\..\..\Results\pca_answer1.txt", "w") as fout:
        fout.write(str(optimal_d))
        
data = pd.read_csv("..\..\..\Data\pca_data_task1.csv")

#%%
from sys import maxint
D = len(data.columns)
best_score = -maxint
best_components_count = 0
for i in xrange(1, D):
    model = PCA(n_components=i)
    scores = cv_score(model, data, cv=3)
    mean_score = np.mean(scores)
    if mean_score > best_score:
        best_score = mean_score
        best_components_count = i

write_answer_1(best_components_count)
print best_components_count
print best_score
