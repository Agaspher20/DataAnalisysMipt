# В приложенном файле — данные по ежемесячному уровню производства молока в фунтах на одну корову.
# Загрузите ряд:
#%%
import pandas as pd
import numpy as np
milk = pd.read_csv("..\..\Data\monthly-milk-production.csv",";", index_col=["month"], parse_dates=["month"], dayfirst=True)
milk.head()

# Создайте новый ряд значений среднего дневного уровня производства молока в фунтах на одну корову, поделив на число дней
# в месяце (вычисляется с помощью функции monthrange из пакета calendar).
#%%
average_day_milk = pd.DataFrame(
    map(lambda (index,row): float(row["milk"])/index.days_in_month, milk.iterrows()),
    milk.index,
    ["milk"])

# Постройте график полученного ряда; какой из приведённых ниже графиков у вас получился?
#%%
average_day_milk.plot()

# Для ряда со средним дневным количеством молока на корову из предыдущего вопроса давайте с помощью критерия Дики-Фуллера
# подберём порядок дифференцирования, при котором ряд становится стационарным.
# Дифференцирование можно делать так:
#%%
diff_average_day_milk = (average_day_milk - average_day_milk.shift(1)).dropna(thresh=1)
# Чтобы сделать сезонное дифференцирование, нужно изменить значение параметра у функции shift:
#%%
diff_average_day_milk_season = (average_day_milk - average_day_milk.shift(12)).dropna(thresh=1)
d2_average_day_milk = (diff_average_day_milk_season - diff_average_day_milk_season.shift(1)).dropna(thresh=1)
# При дифференцировании длина ряда сокращается, поэтому в части строк в новой колонке значения будут не определены (NaN).
# Подавая полученные столбцы на вход критерию Дики-Фуллера, отрезайте неопределённые значения, иначе вы получите
# неопределённый достигаемый уровень значимости.
# Итак, какое дифференцирование делает ряд стационарным?
#%%
import statsmodels.api as sm
(adm_adf,adm_pvalue,adm_usedlag,adm_nobes,adm_icbest,adm_resstore) = sm.tsa.stattools.adfuller(average_day_milk["milk"])
(dadm_adf,dadm_pvalue,dadm_usedlag,dadm_nobes,dadm_icbest,dadm_resstore) = sm.tsa.stattools.adfuller(diff_average_day_milk["milk"])
(sdadm_adf,sdadm_pvalue,sdadm_usedlag,sdadm_nobes,sdadm_icbest,sdadm_resstore) = sm.tsa.stattools.adfuller(diff_average_day_milk_season["milk"])
(d2adm_adf,d2adm_pvalue,d2adm_usedlag,d2adm_nobes,d2adm_icbest,d2adm_resstore) = sm.tsa.stattools.adfuller(d2_average_day_milk["milk"])
print "Initial p-value: %.4f\nDiff p-value: %.4f\nSeason diff p-value: %.4f\nDiff after season diff p-value: %f" % (adm_pvalue, dadm_pvalue, sdadm_pvalue, d2adm_pvalue)

# Для стационарного ряда из предыдущего вопроса (продифференцированного столько раз, сколько вы посчитали нужным) постройте
# график автокорреляционной функции. Это можно cделать так:
#%%
sm.graphics.tsa.plot_acf(d2_average_day_milk.values.squeeze(), lags=50)
# Исходя из этого графика, какое начальное приближение вы предложили бы для параметра Q в модели SARIMA?

# Для того же ряда, что и в предыдущем вопросе, постройте график частичной автокорреляционной функции.
# Это можно сделать так:
#%%
sm.graphics.tsa.plot_pacf(d2_average_day_milk.values.squeeze(), lags=50)
# Исходя из этого графика, какое начальное приближение вы предложили бы для параметра p в модели SARIMA?
