# Для 61 большого города в Англии и Уэльсе известны средняя годовая смертность на 100000 населения
# (по данным 1958–1964) и концентрация кальция в питьевой воде (в частях на миллион). Чем выше концентрация
# кальция, тем жёстче вода. Города дополнительно поделены на северные и южные.

# Постройте 95% доверительный интервал для средней годовой смертности в больших городах. Чему равна его нижняя
# граница? Округлите ответ до 4 знаков после десятичной точки. 

#%%
import pandas as pd
import numpy as np

frame = pd.read_csv("..\..\Data\water.txt", sep="\t", header=0)

#%%
print frame.shape
frame.head()

#%%
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic

def confidential_bounds(frame):
    std = frame.std(ddof=1)
    mean = frame.mean()
    count = frame.shape[0]
    return _tconfint_generic(mean, std/np.sqrt(count), count-1, 0.05, "two-sided")
#%%
trust_interval_mortality = confidential_bounds(frame["mortality"])
print "Mortality trust interval: ", np.round(trust_interval_mortality, 4)
print "Mortality mean", np.round(frame["mortality"].mean(), 4)

# На данных из предыдущего вопроса постройте 95% доверительный интервал для средней годовой смертности по всем
# южным городам. Чему равна его верхняя граница? Округлите ответ до 4 знаков после десятичной точки.
#%%
south_frame = frame[frame["location"] == "South"]
trust_interval_mortality_south = confidential_bounds(south_frame["mortality"])
print "Mortality for south cities trust interval: ", np.round(trust_interval_mortality_south, 4)
print "Mortality for south cities mean: ", np.round(south_frame["mortality"].mean(), 4)

# На тех же данных постройте 95% доверительный интервал для средней годовой смертности по всем северным городам.
# Пересекается ли этот интервал с предыдущим? Как вы думаете, какой из этого можно сделать вывод? 
#%%
north_frame = frame[frame["location"] == "North"]
trust_interval_mortality_north = confidential_bounds(north_frame["mortality"])
print "Mortality for north cities trust interval: ", np.round(trust_interval_mortality_north, 4)
print "Mortality for north cities mean: ", np.round(north_frame["mortality"].mean(), 4)
# Интервалы не пересекаются; видимо, средняя смертность на севере и на юге существенно разная 

# Пересекаются ли 95% доверительные интервалы для средней жёсткости воды в северных и южных городах?
#%%
trust_interval_hardness_south = confidential_bounds(south_frame["hardness"])
trust_interval_hardness_north = confidential_bounds(north_frame["hardness"])
print "Water hardness trust interval for south cities: ", np.round(trust_interval_hardness_south, 4)
print "Water hardness trust interval for north cities: ", np.round(trust_interval_hardness_north, 4)
# Не пересекаются

# Вспомним формулу доверительного интервала для среднего нормально распределённой случайной величины
# с дисперсией sigma^2
# При σ=1 какой нужен объём выборки, чтобы на уровне доверия 95% оценить среднее с точностью ±0.1?
#%%
z = 1.95996
print (1./(0.1/z))**2
