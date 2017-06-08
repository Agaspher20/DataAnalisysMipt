# В одном из выпусков программы "Разрушители легенд" проверялось, действительно ли заразительна зевота.
# В эксперименте участвовало 50 испытуемых, проходивших собеседование на программу. Каждый из них разговаривал
# с рекрутером; в конце 34 из 50 бесед рекрутер зевал. Затем испытуемых просили подождать решения рекрутера в
# соседней пустой комнате.
# Во время ожидания 10 из 34 испытуемых экспериментальной группы и 4 из 16 испытуемых контрольной начали зевать.
# Таким образом, разница в доле зевающих людей в этих двух группах составила примерно 4.4%. Ведущие заключили,
# что миф о заразительности зевоты подтверждён.
# Можно ли утверждать, что доли зевающих в контрольной и экспериментальной группах отличаются статистически
# значимо? Посчитайте достигаемый уровень значимости при альтернативе заразительности зевоты, округлите до
# четырёх знаков после десятичной точки.
#%%
import numpy as np
import scipy
from statsmodels.stats.weightstats import *
from statsmodels.stats.proportion import proportion_confint

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
print "p-value: %f" % proportions_diff_z_test(proportions_diff_z_stat_ind(zev_vector, ctrl_vector))
