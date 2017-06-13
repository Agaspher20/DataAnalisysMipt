#%%
import numpy as np
import scipy
from statsmodels.stats.weightstats import *
from statsmodels.stats.proportion import proportion_confint

# В одном из выпусков программы "Разрушители легенд" проверялось, действительно ли заразительна зевота.
# В эксперименте участвовало 50 испытуемых, проходивших собеседование на программу. Каждый из них разговаривал
# с рекрутером; в конце 34 из 50 бесед рекрутер зевал. Затем испытуемых просили подождать решения рекрутера в
# соседней пустой комнате.
# Во время ожидания 10 из 34 испытуемых экспериментальной группы и 4 из 16 испытуемых контрольной начали зевать.
# Таким образом, разница в доле зевающих людей в этих двух группах составила примерно 4.4%. Ведущие заключили,
# что миф о заразительности зевоты подтверждён.
#%%
n = 50
n_zev = 34
n_ctrl = 50-34
n_zev_zev = 10
n_ctrl_zev = 4

zev_vector = np.append(
    np.ones(int(n_zev_zev), dtype=int),
    np.zeros(int(n_zev-n_zev_zev), dtype=int))
ctrl_vector = np.append(
    np.ones(int(n_ctrl_zev), dtype=int),
    np.zeros(int(n_ctrl-n_ctrl_zev), dtype=int)
)

print "Difference: %.4f" % np.round(float(n_zev_zev)/float(n_zev) - float(n_ctrl_zev)/float(n_ctrl), 4)

# Можно ли утверждать, что доли зевающих в контрольной и экспериментальной группах отличаются статистически
# значимо? Посчитайте достигаемый уровень значимости при альтернативе заразительности зевоты, округлите до
# четырёх знаков после десятичной точки.
#%%
conf_interval_zev = proportion_confint(
    sum(zev_vector),
    zev_vector.shape[0],
    method = 'wilson')
conf_interval_ctrl = proportion_confint(
    sum(ctrl_vector), 
    ctrl_vector.shape[0],
    method = 'wilson')

print '95%% confidence interval for a zev probability, zev group: [%f, %f]' % conf_interval_zev
print '95%% confidence interval for a zev probability, control group: [%f, %f]' % conf_interval_ctrl

#%%
def proportions_diff_confint_ind(sample1, sample2, alpha = 0.05):    
    z = scipy.stats.norm.ppf(1 - alpha / 2.)
    
    p1 = float(sum(sample1)) / len(sample1)
    p2 = float(sum(sample2)) / len(sample2)
    
    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1)/ len(sample1) + p2 * (1 - p2)/ len(sample2))
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1)/ len(sample1) + p2 * (1 - p2)/ len(sample2))
    
    return (left_boundary, right_boundary)

def proportions_diff_z_stat_ind(sample1, sample2):
    n1 = len(sample1)
    n2 = len(sample2)
    
    p1 = float(sum(sample1)) / n1
    p2 = float(sum(sample2)) / n2 
    P = float(p1*n1 + p2*n2) / (n1 + n2)
    
    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))

def proportions_diff_z_test(z_stat, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")
    
    if alternative == 'two-sided':
        return 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))
    
    if alternative == 'less':
        return scipy.stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - scipy.stats.norm.cdf(z_stat)

#%%
print "95%% confidence interval for a difference between proportions: [%f, %f]" %\
      proportions_diff_confint_ind(zev_vector, ctrl_vector)

#%%
pvalue = proportions_diff_z_test(proportions_diff_z_stat_ind(zev_vector, ctrl_vector), alternative='greater')
print "p-value: %.4f" % np.round(pvalue, 4)

# Имеются данные измерений двухсот швейцарских тысячефранковых банкнот, бывших в обращении в первой половине XX века.
# Сто из банкнот были настоящими, и сто — поддельными. На рисунке ниже показаны измеренные признаки:
# Загрузите данные.
#%%
import pandas as pd
frame = pd.read_csv("..\..\Data\\banknotes.txt", sep="\t", header=0)
frame.head()

# Отделите 50 случайных наблюдений в тестовую выборку с помощью функции sklearn.cross_validation.train_test_split (зафиксируйте random state = 1).
#%%
from sklearn.model_selection import train_test_split
y = frame["real"].as_matrix()
x = frame.drop("real", axis=1).as_matrix()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=50, random_state=1)

# На оставшихся 150 настройте два классификатора поддельности банкнот:
#  1. логистическая регрессия по признакам X1,X2,X3;
#  2. логистическая регрессия по признакам X4,X5,X6.
#%%
from sklearn.linear_model import LogisticRegression
x_train_1_3 = x_train[np.ix_(range(0, x_train.shape[0]), [0,1,2])]
x_test_1_3 = x_test[np.ix_(range(0, x_test.shape[0]), [0,1,2])]

x_train_4_6 = x_train[np.ix_(range(0, x_train.shape[0]), [3,4,5])]
x_test_4_6 = x_test[np.ix_(range(0, x_test.shape[0]), [3,4,5])]

regression_1_3 = LogisticRegression().fit(x_train_1_3, y_train)
regression_4_6 = LogisticRegression().fit(x_train_4_6, y_train)

# Каждым из классификаторов сделайте предсказания меток классов на тестовой выборке. Одинаковы ли доли ошибочных предсказаний двух классификаторов?
#%%
predictions_1_3 = map(lambda x_row: regression_1_3.predict(x_row.reshape(1,-1))[0], x_test_1_3)
predictions_4_6 = map(lambda x_row: regression_4_6.predict(x_row.reshape(1,-1))[0], x_test_4_6)

predictions_res_1_3 = map(lambda (pred, val): 1 if pred == val else 0, zip(predictions_1_3, y_test))
predictions_res_4_6 = map(lambda (pred, val): 1 if pred == val else 0, zip(predictions_4_6, y_test))

print "False predictions portion 1-3: %f" % (1.-float(sum(predictions_res_1_3))/float(len(predictions_res_1_3)))
print "False predictions portion 4-6: %f" % (1.-float(sum(predictions_res_4_6))/float(len(predictions_res_4_6)))

# Проверьте гипотезу, вычислите достигаемый уровень значимости. Введите номер первой значащей цифры (например, если вы получили 5.5×10−8, нужно ввести 8).
#%%
def proportions_diff_confint_rel(sample1, sample2, alpha = 0.05):
    z = scipy.stats.norm.ppf(1 - alpha / 2.)
    sample = zip(sample1, sample2)
    n = len(sample)
        
    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])
    
    left_boundary = float(f - g) / n  - z * np.sqrt(float((f + g)) / n**2 - float((f - g)**2) / n**3)
    right_boundary = float(f - g) / n  + z * np.sqrt(float((f + g)) / n**2 - float((f - g)**2) / n**3)
    return (left_boundary, right_boundary)

def proportions_diff_z_stat_rel(sample1, sample2):
    sample = zip(sample1, sample2)
    n = len(sample)
    
    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])
    
    return float(f - g) / np.sqrt(f + g - float((f - g)**2) / n )
#%%
pvalue = proportions_diff_z_test(proportions_diff_z_stat_rel(predictions_res_1_3, predictions_res_4_6))
print "p-value: %f" % proportions_diff_z_test(proportions_diff_z_stat_rel(predictions_res_1_3, predictions_res_4_6))

# В предыдущей задаче посчитайте 95% доверительный интервал для разности долей ошибок двух классификаторов. Чему равна его ближайшая к нулю граница?
# Округлите до четырёх знаков после десятичной точки.
#%%
cinfidence_int_rel = proportions_diff_confint_rel(predictions_res_1_3, predictions_res_4_6)
print "95%% confidence interval for a difference between predictions: [%.4f, %.4f]" \
      % (np.round(cinfidence_int_rel[0],4), np.round(cinfidence_int_rel[1], 4))

# Ежегодно более 200000 людей по всему миру сдают стандартизированный экзамен GMAT при поступлении на программы MBA.
# Средний результат составляет 525 баллов, стандартное отклонение — 100 баллов.
# Сто студентов закончили специальные подготовительные курсы и сдали экзамен. Средний полученный ими балл — 541.4.
#%%
n = 200000.
miu = 525.
std = 100.
n_yr = 100.
mean_yr = 541.4

# Проверьте гипотезу о неэффективности программы против односторонней альтернативы о том, что программа работает.
# Отвергается ли на уровне значимости 0.05 нулевая гипотеза? Введите достигаемый уровень значимости, округлённый до 4 знаков после десятичной точки.
#%%
from scipy.stats import norm
def check_hypothesis(mean, n, miu, std):
    Z = (mean - miu)/(std/np.sqrt(n))
    return (Z, 1.-norm.cdf(Z))
#%%
print np.round(check_hypothesis(mean_yr, n_yr, miu, std), 4)

# Оцените теперь эффективность подготовительных курсов, средний балл 100 выпускников которых равен 541.5.
# Отвергается ли на уровне значимости 0.05 та же самая нулевая гипотеза против той же самой альтернативы?
# Введите достигаемый уровень значимости, округлённый до 4 знаков после десятичной точки. 
#%%
mean_yr = 541.5
print np.round(check_hypothesis(mean_yr, n_yr, miu, std),4)

