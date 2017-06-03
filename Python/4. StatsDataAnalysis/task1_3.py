#%%
import numpy as np
from scipy.stats import norm
# Давайте уточним правило трёх сигм.
# Утверждение: 99.7% вероятностной массы случайной величины X∼N(μ,σ^2) лежит в интервале μ±c⋅σ.
# Чему равно точное значение константы c?
# Округлите ответ до четырёх знаков после десятичной точки. 
#%%
print "3 sigma rule refined: %.4f" % np.round(norm.ppf(0.003/2), 4)

# В пятилетнем рандомизированном исследовании Гарвардской медицинской школы 11037 испытуемых через день
# принимали аспирин, а ещё 11034 — плацебо. Исследование было слепым, то есть, испытуемые не знали,
# что именно они принимают.
# За 5 лет инфаркт случился у 104 испытуемых, принимавших аспирин, и у 189 принимавших плацебо.
# Оцените, насколько вероятность инфаркта снижается при приёме аспирина. Округлите ответ до четырёх знаков
# после десятичной точки.
#%%
aspirin_n = 11037
placebo_n = 11034
aspirin_infarct_n = 104
placebo_infarct_n = 189

aspirin_p = float(aspirin_infarct_n)/aspirin_n
placebo_p = float(placebo_infarct_n)/placebo_n

print "Aspirin infarct probability: %.4f" % aspirin_p
print "Placebo infarct probability: %.4f" % placebo_p
print "Probability diff %.4f" % np.round(placebo_p-aspirin_p, 4)

#%%
from statsmodels.stats.proportion import proportion_confint
conf_interval_aspirin = proportion_confint(aspirin_infarct_n, aspirin_n, method="wilson")
conf_interval_placebo = proportion_confint(placebo_infarct_n, placebo_n, method="wilson")

print "interval for aspirin infarct [%.4f, %.4f]" % (np.round(conf_interval_aspirin[0], 4), np.round(conf_interval_aspirin[1], 4))
print "interval for placebo infarct [%.4f, %.4f]" % (np.round(conf_interval_placebo[0], 4), np.round(conf_interval_placebo[1], 4))

#%%
def proportions_confint_diff_ind(p1, count1, p2, count2, alpha = 0.05):    
    z = norm.ppf(1 - alpha / 2.)
    
    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1. - p1)/ count1 + p2 * (1 - p2)/ count2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1. - p1)/ count1 + p2 * (1 - p2)/ count2)
    
    return (left_boundary, right_boundary)

# Постройте теперь 95% доверительный интервал для снижения вероятности инфаркта при приёме аспирина.
# Чему равна его верхняя граница? Округлите ответ до четырёх знаков после десятичной точки.
#%%
conf_interval_diff = proportions_confint_diff_ind(placebo_p, placebo_n, aspirin_p, aspirin_n)
print "interval for aspirin and placebo probability diff: [%.4f, %.4f]" % (np.round(conf_interval_diff[0],4),np.round(conf_interval_diff[1],4))

# Продолжим анализировать данные эксперимента Гарвардской медицинской школы.
# Для бернуллиевских случайных величин X∼Ber(p) часто вычисляют величину p/(1−p), которая называется шансами
# (odds). Чтобы оценить шансы по выборке, вместо p нужно подставить p^. Например, шансы инфаркта в контрольной
# группе, принимавшей плацебо, можно оценить как
# (189/11034)/(1−189/11034)=189/(11034−189)≈0.0174
# Оцените, во сколько раз понижаются шансы инфаркта при регулярном приёме аспирина. Округлите ответ до четырёх
# знаков после десятичной точки. 
#%%
def calculate_odds(success_count, full_count):
    return float(success_count)/float(full_count-success_count)
#%%
aspirin_odds = calculate_odds(aspirin_infarct_n, aspirin_n)
placebo_odds = calculate_odds(placebo_infarct_n, placebo_n)
print "Placebo to aspirin odds ratio: %.4f" % np.round(placebo_odds/aspirin_odds, 4)

# Величина, которую вы оценили в предыдущем вопросе, называется отношением шансов. Постройте для отношения
# шансов 95% доверительный интервал с помощью бутстрепа. Чему равна его нижняя граница?
# Округлите ответ до 4 знаков после десятичной точки.

# Чтобы получить в точности такой же доверительный интервал, как у нас:
# - составьте векторы исходов в контрольной и тестовой выборках так, чтобы в начале шли все единицы,
#   а потом все нули;
# - установите random seed=0;
# - сделайте по 1000 псевдовыборок из каждой группы пациентов с помощью функции get_bootstrap_samples.
#%%
aspirin_vector = np.append(
    np.ones(int(aspirin_infarct_n), dtype=int),
    np.zeros(int(aspirin_n-aspirin_infarct_n), dtype=int))
placebo_vector = np.append(
    np.ones(int(placebo_infarct_n), dtype=int),
    np.zeros(int(placebo_n-placebo_infarct_n), dtype=int)
)
#%%
def get_bootstrap_samples(data, n_samples):
    data_length = len(data)
    indices = np.random.randint(0, data_length, (n_samples, data_length))
    return data[indices]
#%%
np.random.seed(0)
aspirin_samples = get_bootstrap_samples(aspirin_vector, 1000)
placebo_samples = get_bootstrap_samples(placebo_vector, 1000)
#%%
def calculate_odds_from_samples(samples):
    samples_count = len(samples)
    success_count = len(samples[samples==1])
    return calculate_odds(success_count, samples_count)
#%%
all_aspirin_odds = map(calculate_odds_from_samples, aspirin_samples)
all_placebo_odds = map(calculate_odds_from_samples, placebo_samples)

#%%
def stat_intervals(stat, alpha):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries
#%%
odds = map(
    lambda (aspirin_odd, placebo_odd): placebo_odd/aspirin_odd,
    zip(all_aspirin_odds,all_placebo_odds))
#%%
print np.round(stat_intervals(odds, 0.05), 4)
