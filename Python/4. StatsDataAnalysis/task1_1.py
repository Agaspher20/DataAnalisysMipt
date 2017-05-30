# Для 61 большого города в Англии и Уэльсе известны средняя годовая смертность на 100000 населения
# (по данным 1958–1964) и концентрация кальция в питьевой воде (в частях на миллион). Чем выше концентрация
# кальция, тем жёстче вода. Города дополнительно поделены на северные и южные.

# Постройте 95% доверительный интервал для средней годовой смертности в больших городах. Чему равна его нижняя
# граница? Округлите ответ до 4 знаков после десятичной точки. 

#%%
import pandas as pd

frame = pd.read_csv("..\..\Data\water.txt", sep="\t", header=0)

#%%
print frame.head()
print frame.shape

#%%
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic

def confidential_bounds(frame):
    std = frame.std(ddof=1)
    mean = frame.mean()
    count = frame.shape[0]
#    return _zconfint_generic(mean, std/np.sqrt(count), 0.05, "two-sided")
    return _tconfint_generic(mean, std/np.sqrt(count), count-1, 0.05, "two-sided")
#%%
minVal,maxVal = confidential_bounds(frame["mortality"])
print np.round(minVal, 4), np.round(maxVal, 4), frame["mortality"].mean()

#%%
south_frame = frame[frame.location=="South"]
south_minVal,south_maxVal = confidential_bounds(south_frame["mortality"])
print np.round(south_minVal, 4), np.round(south_maxVal, 4), south_frame["mortality"].mean()

#%%
north_frame = frame[frame.location=="North"]
north_minVal,north_maxVal = confidential_bounds(north_frame["mortality"])
print np.round(north_minVal, 4), np.round(north_maxVal, 4)

#%%
south_hardness_min,south_hardness_max = confidential_bounds(south_frame["hardness"])
north_hardness_min,north_hardness_max = confidential_bounds(north_frame["hardness"])
print np.round(south_hardness_min, 4), np.round(south_hardness_max, 4)
print np.round(north_hardness_min, 4), np.round(north_hardness_max, 4)

#%%
z = 1.95996
print (1./(0.1/z))**2