# Уровень кальция в крови здоровых молодых женщин равен в среднем 9.5 милиграммам на децилитр и имеет
# характерное стандартное отклонение 0.4 мг/дл. В сельской больнице Гватемалы для 160 здоровых беременных женщин
# при первом обращении для ведения беременности был измерен уровень кальция; среднее значение составило
# 9.57 мг/дл. Можно ли утверждать, что средний уровень кальция в этой популяции отличается от 9.5?
# 
# Посчитайте достигаемый уровень значимости. Поскольку известны только среднее и дисперсия, а не сама выборка,
# нельзя использовать стандартные функции критериев — нужно реализовать формулу достигаемого уровня значимости
# самостоятельно.
# 
# Округлите ответ до четырёх знаков после десятичной точки.
#%%
import numpy as np
from scipy.stats import norm
avg = 9.5
std = 0.4
n = 160
avg_sel = 9.57
Z = (avg_sel - avg)/(std/np.sqrt(n))
p_val = 2.*(1.-norm.cdf(Z))
print np.round((Z,p_val,norm.cdf(Z)), 4)

# Имеются данные о стоимости и размерах 53940 бриллиантов.
#%%
import pandas as pd
frame = pd.read_csv("diamonds.txt", sep="\t", header=0)
frame.head()

# Отделите 25% случайных наблюдений в тестовую выборку с помощью функции
# sklearn.cross_validation.train_test_split (зафиксируйте random state = 1).
#%%
from sklearn.cross_validation import train_test_split
y = frame["price"].as_matrix()
x = frame.drop("price", axis=1).as_matrix()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# На обучающей выборке настройте две регрессионные модели:
#     1. линейную регрессию с помощью LinearRegression без параметров
#     2. случайный лес с помощью RandomForestRegressor с random_state=1.
#%%
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
linear = LinearRegression().fit(x_train, y_train)
forest = RandomForestRegressor(random_state=1).fit(x_train, y_train)

# Какая из моделей лучше предсказывает цену бриллиантов?
# Сделайте предсказания на тестовой выборке.
#%%
forest_predictions = map(lambda x_row: forest.predict(x_row.reshape(1,-1))[0], x_test)
linear_predictions = map(lambda x_row: linear.predict(x_test[0].reshape(1,-1))[0], x_test)

# Посчитайте модули отклонений предсказаний от истинных цен.
#%%
forest_abs_devs = np.array(map(lambda (prediction,y_val): np.abs(prediction-y_val), zip(forest_predictions,y_test)))
linear_abs_devs = np.array(map(lambda (prediction,y_val): np.abs(prediction-y_val), zip(linear_predictions,y_test)))

# Проверьте гипотезу об одинаковом среднем качестве предсказаний.
#%%
from statsmodels.stats.weightstats import *
%pylab inline
stats.probplot(forest_abs_devs-linear_abs_devs, dist = "norm", plot = pylab)
pylab.show()
#%%
print "Shapiro-Wilk normality test, W-statistic: %f, p-value: %f" % stats.shapiro(forest_abs_devs-linear_abs_devs)
# Вычислите достигаемый уровень значимости.
# Отвергается ли гипотеза об одинаковом качестве моделей против двусторонней альтернативы на уровне значимости
# α=0.05?
#%%
stats.ttest_rel(linear_abs_devs, forest_abs_devs)

# В предыдущей задаче посчитайте 95% доверительный интервал для разности средних абсолютных ошибок предсказаний
# регрессии и случайного леса. Чему равна его ближайшая к нулю граница? Округлите до десятков (поскольку
# случайный лес может давать немного разные предсказания в зависимости от версий библиотек, мы просим вас так
# сильно округлить, чтобы полученное значение наверняка совпало с нашим).
#%%
print "95%% confidence interval: [%f, %f]" % DescrStatsW(linear_abs_devs - forest_abs_devs).tconfint_mean()