# Давайте вернёмся к данным выживаемости пациентов с лейкоцитарной лимфомой из видео про критерий знаков:
# Измерено остаточное время жизни с момента начала наблюдения (в неделях);
# звёздочка обозначает цензурирование сверху — исследование длилось 7 лет, и остаточное время жизни одного пациента, который дожил до конца наблюдения, неизвестно.
#%%
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import zconfint

life_times = np.array([49,58,75,110,112,132,151,276,281,362]) #∗
print "95%% confidence interval for the life time: [%f, %f]" % zconfint(life_times)
# Поскольку цензурировано только одно наблюдение, для проверки гипотезы H0:medX=200 на этих данных можно использовать критерий знаковых рангов — можно считать,
# что время дожития последнего пациента в точности равно 362, на ранг этого наблюдения это никак не повлияет.
# Критерием знаковых рангов проверьте эту гипотезу против двусторонней альтернативы, выведите достигаемый уровень значимости,
# округлённый до четырёх знаков после десятичной точки.
#%%
medX = 200
print "Wilcoxon criterion pvalue result: %.4f" % np.round(stats.wilcoxon(life_times-medX).pvalue, 4)

#%%
# В ходе исследования влияния лесозаготовки на биоразнообразие лесов острова Борнео собраны данные о количестве видов деревьев в 12 лесах, где вырубка не ведётся:
no_cut_kinds = np.array([22,22,15,13,19,19,18,20,21,13,13,15])
# и в 9 лесах, где идёт вырубка:
cut_kinds = np.array([17,18,18,15,12,4,14,15,10])
# Проверьте гипотезу о равенстве среднего количества видов в двух типах лесов против односторонней альтернативы о снижении биоразнообразия в вырубаемых лесах.
# Используйте ранговый критерий. Чему равен достигаемый уровень значимости? Округлите до четырёх знаков после десятичной точки.
#%%
print "Mann-Whitney criterion pvalue result: %.4f" % np.round(stats.mannwhitneyu(no_cut_kinds, cut_kinds, alternative="greater").pvalue, 4)

# 28 января 1986 года космический шаттл "Челленджер" взорвался при взлёте. Семь астронавтов, находившихся на борту, погибли.
# В ходе расследования причин катастрофы основной версией была неполадка с резиновыми уплотнительными кольцами в соединении с ракетными ускорителями.
# Для 23 предшествовавших катастрофе полётов "Челленджера" известны температура воздуха и появление повреждений хотя бы у одного из уплотнительных колец.
#%%
import pandas as pd
challenger_frame = pd.read_csv("..\..\Data\\challenger.txt", sep="\t", header=0)
challenger_frame.head()
# С помощью бутстрепа постройте 95% доверительный интервал для разности средних температур воздуха при запусках, когда уплотнительные кольца повреждались,
# и запусках, когда повреждений не было. Чему равна его ближайшая к нулю граница? Округлите до четырёх знаков после запятой.
#%%
temprature_incident = challenger_frame[challenger_frame["Incident"] == 1]["Temperature"].as_matrix()
temprature_ok = challenger_frame[challenger_frame["Incident"] == 0]["Temperature"].as_matrix()
print "Incidents count: %i" % temprature_incident.shape[0]
print "Ok count: %i" % temprature_ok.shape[0]
# Чтобы получить в точности такой же доверительный интервал, как у нас:
# установите random seed = 0 перед первым вызовом функции get_bootstrap_samples, один раз
# сделайте по 1000 псевдовыборок из каждой выборки.
#%%
def get_bootstrap_samples(data, n_samples):
    data_length = len(data)
    indices = np.random.randint(0, data_length, (n_samples, data_length))
    return data[indices]
def stat_intervals(stat, alpha):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries
#%%
samples_count = 1000
np.random.seed(0)
incident_selection = map(lambda temps: np.mean(temps), get_bootstrap_samples(temprature_incident, samples_count))
ok_selection = map(lambda temps: np.mean(temps), get_bootstrap_samples(temprature_ok, samples_count))
temprature_differencies = map(lambda (ok_temp,inc_temp): ok_temp-inc_temp, zip(ok_selection,incident_selection))
print np.round(stat_intervals(temprature_differencies, 0.05), 4)

# На данных предыдущей задачи проверьте гипотезу об одинаковой средней температуре воздуха в дни, когда уплотнительный кольца повреждались,
# и дни, когда повреждений не было. Используйте перестановочный критерий и двустороннюю альтернативу.
#%%
def permutation_t_stat_ind(sample1, sample2):
    return np.mean(sample1) - np.mean(sample2)
def get_random_combinations(n1, n2, max_combinations):
    index = range(n1 + n2)
    indices = set([tuple(index)])
    for i in range(max_combinations - 1):
        np.random.shuffle(index)
        indices.add(tuple(index))
    return [(index[:n1], index[n1:]) for index in indices]
def permutation_zero_dist_ind(sample1, sample2, max_combinations = None):
    joined_sample = np.hstack((sample1, sample2))
    n1 = len(sample1)
    n = len(joined_sample)
    
    if max_combinations:
        indices = get_random_combinations(n1, len(sample2), max_combinations)
    else:
        indices = [(list(index), filter(lambda i: i not in index, range(n))) \
                    for index in itertools.combinations(range(n), n1)]
    
    distr = [joined_sample[list(i[0])].mean() - joined_sample[list(i[1])].mean() \
             for i in indices]
    return distr
def permutation_test(sample, mean, max_permutations = None, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")
    
    t_stat = permutation_t_stat_ind(sample, mean)
    
    zero_distr = permutation_zero_dist_ind(sample, mean, max_permutations)
    
    if alternative == 'two-sided':
        return sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)
    
    if alternative == 'less':
        return sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        return sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)

# Чему равен достигаемый уровень значимости? Округлите до четырёх знаков после десятичной точки.
# Чтобы получить такое же значение, как мы:
#   1. установите random seed = 0;
#   2. возьмите 10000 перестановок.
#%%
permutations_count = 10000
np.random.seed(0)
print "p-value: %.4f" % np.round(permutation_test(temprature_ok, temprature_incident, max_permutations=permutations_count, alternative="two-sided"), 4)
