# Есть ли связь между неграмотностью и рождаемостью? Для 94 стран, уровень неграмотности женщин в
# которых больше 5%, известны доля неграмотных среди женщин старше 15 (на 2003 год) и средняя
# рождаемость на одну женщину (на 2005 год).
#%%
import pandas as pd
import numpy as np
frame = pd.read_csv("illiteracy.txt", sep="\t", header=0)
frame.head()
# Чему равен выборочный коэффициент корреляции Пирсона между этими двумя признаками?
# Округлите до четырёх знаков после десятичной точки.
#%%
print "Pearson correlation Illiteracy-Births: %.4f" % np.round(frame.corr(method="pearson")["Illit"]["Births"], 4)
print "Spearman correlation Illiteracy-Births: %.4f" % np.round(frame.corr(method="spearman")["Illit"]["Births"], 4)
