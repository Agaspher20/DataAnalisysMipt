# В данном задании вам нужно будет
#    проанализировать АБ тест, проведенный на реальных пользователях Яндекса
#    подтвердить или опровергнуть наличие изменений в пользовательском поведении между контрольной (control)
#       и тестовой (exp) группами
#    определить характер этих изменений и практическую значимость вводимого изменения
#    понять, какая из пользовательских групп более всего проигрывает / выигрывает от тестируемого изменения
#       (локализовать изменение)
# Описание данных:
#     userID: уникальный идентификатор пользователя
#     browser: браузер, который использовал userID
#     slot: в каком статусе пользователь участвовал в исследовании
#       (exp = видел измененную страницу, control = видел неизменную страницу)
#     n_clicks: количество кликов, которые пользоваль совершил за n_queries
#     n_queries: количество запросов, который совершил userID, пользуясь браузером browser
#     n_nonclk_queries: количество запросов пользователя, в которых им не было совершено ни одного клика
# Обращаем ваше внимание, что не все люди используют только один браузер, поэтому в столбце userID есть
# повторяющиеся идентификаторы. В предлагаемых данных уникальным является сочетание userID и browser.
#%%
import pandas as pd
import numpy as np

frame = pd.read_csv("ab_browser_test.csv", sep=",", header=0)
frame.head()

# Основная метрика, на которой мы сосредоточимся в этой работе, — это количество пользовательских кликов на
# web-странице в зависимости от тестируемого изменения этой страницы.
# Посчитайте, насколько в группе exp больше пользовательских кликов по сравнению с группой control в процентах
# от числа кликов в контрольной группе.
# Полученный процент округлите до третьего знака после точки.
#%%
exp_clicks_count = frame[frame["slot"]=="exp"]["n_clicks"].sum()
control_clicks_count = frame[frame["slot"]=="control"]["n_clicks"].sum()
print "Exp clicks count difference percent: %.3f" % np.round(float(exp_clicks_count)*100./float(control_clicks_count)-100, 3)

# Давайте попробуем посмотреть более внимательно на разницу между двумя группами (control и exp) относительно
# количества пользовательских кликов.
# Для этого постройте с помощью бутстрепа 95% доверительный интервал для средних значений и медиан количества
# кликов в каждой из двух групп. Отметьте все верные утверждения.
#%%
def get_bootstrap_samples(data, n_samples):
    data_length = len(data)
    indices = np.random.randint(0, data_length, (n_samples, data_length))
    return data[indices]

def stat_intervals(stat, alpha):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries
#%%
np.random.seed(0)
n_boot_samples = 500
exp_clicks = get_bootstrap_samples(frame[frame["slot"]=="exp"]["n_clicks"].values, n_boot_samples)
control_clicks = get_bootstrap_samples(frame[frame["slot"]=="control"]["n_clicks"].values, n_boot_samples)

exp_clicks_means = map(np.mean, exp_clicks)
exp_clicks_medians = map(np.median, exp_clicks)

control_clicks_means = map(np.mean, control_clicks)
control_clicks_medians = map(np.median, control_clicks)

exp_means_interval = stat_intervals(exp_clicks_means, 0.05)
exp_medians_interval = stat_intervals(exp_clicks_medians, 0.05)

control_means_interval = stat_intervals(control_clicks_means, 0.05)
control_medians_interval = stat_intervals(control_clicks_medians, 0.05)

interval_means_diff = stat_intervals(map(lambda(x,y): x-y, zip(exp_clicks_means,control_clicks_means)), 0.05)
interval_medians_diff = stat_intervals(map(lambda(x,y): x-y, zip(exp_clicks_medians,control_clicks_medians)), 0.05)

print "95%% Confidence interval for exp means: [%f;%f]" % tuple(exp_means_interval)
print "95%% Confidence interval for control means: [%f;%f]" % tuple(control_means_interval)
print "95%% Confidence interval for exp medians: [%f;%f]" % tuple(exp_medians_interval)
print "95%% Confidence interval for control medians: [%f;%f]" % tuple(control_medians_interval)
print "95%% Confidence interval for difference between means: [%f;%f]" % tuple(interval_means_diff)
print "95%% Confidence interval for difference between medians: [%f;%f]" % tuple(interval_medians_diff)

# 95% доверительный интервал для разности средних не содержит ноль, похоже, средние отличаются статистически значимо
# 95% доверительный интервал для разности медиан не содержит ноль, похоже, медианы отличаются статистически значимо

# Поскольку данных достаточно много (порядка полумиллиона уникальных пользователей), отличие в несколько процентов
# может быть не только практически значимым, но и значимым статистически.
# Последнее утверждение нуждается в дополнительной проверке.
# Посмотрите на выданные вам данные и выберите все верные варианты ответа относительно проверки гипотезы о равенстве
#  среднего количества кликов в группах.
# Все ответы не верны

# t-критерий Стьюдента имеет множество достоинств, и потому его достаточно часто применяют в AB экспериментах.
# Иногда его применение может быть необоснованно из-за сильной скошенности распределения данных.
# Давайте постараемся понять, когда t-критерий можно применять и как это проверить на реальных данных.
# Для простоты рассмотрим одновыборочный t-критерий. Его статистика имеет вид (X¯−μ)/sqrt(S^2/n'), то есть чтобы
# действительно предположения t-критерия выполнялись необходимо, чтобы:
#   X¯ — среднее значение в выборке — было распределено нормально N(μ,σ2n)
#   (n/σ^2)S^2 — несмещенная оценка дисперсии c масштабирующим коэффициентом — была распределена по хи-квадрат c n−1
#       степенями свободы χ^2(n−1) 

# Простое доказательство необходимости и достаточности этого требования можно посмотреть в самом последнем абзаце
# этого вопроса.
# Усвоение этого доказательства не обязательно для выполнения задания.

# Оба этих предположения можно проверить с помощью бутстрепа. Ограничимся сейчас только контрольной группой,
# в которой распределение кликов будем называть данными в рамках данного вопроса.
# Поскольку мы не знаем истинного распределения генеральной совокупности, мы можем применить бутстреп, чтобы
# понять, как распределены среднее значение и выборочная дисперсия. Для этого
#   Получите из данных n_boot_samples псевдовыборок.
#   По каждой из этих выборок посчитайте среднее и сумму квадратов отклонения от выборочного среднего
#       (control_boot_chi_squared)
#   Для получившегося вектора средних значений из n_boot_samples постройте q-q plot с помощью
#       scipy.stats.probplot для нормального распределения
#   Для получившегося вектора сумм квадратов отклонения от выборочного среднего постройте qq-plot с помощью
#       scipy.stats.probplot для хи-квадрат распределения с помощью команды 
#            scipy.stats.probplot(control_boot_chi_squared, dist="chi2", 
#                                sparams=(n-1), plot=plt)
# Где sparams=(n-1) означают число степеней свободы = длине выборки - 1.
# Чтобы получить такой же ответ, как у нас, зафиксируйте seed и количество псевдовыборок:
#%%
from scipy import stats
%pylab inline
sample_0 = control_clicks[0]
expected_frequences_s0 = [len(sample_0)*stats.norm.pdf(x) for x in range(min(sample_0), max(sample_0) + 1)]
pylab.xlim(0, 50)
pylab.bar(range(len(np.bincount(sample_0))), np.bincount(sample_0), color = 'b', label = 'sample0_stat')
pylab.bar(range(len(expected_frequences_s0)), expected_frequences_s0, color = 'r', label = 'norm_distr')
pylab.legend()
#%%
from scipy import stats
def calc_chisquare_stats(sample):
    mean = np.mean(sample)
    sample_size = len(sample)
    expected_means = [mean for x in xrange(0, sample_size)]
    stat = stats.chisquare(sample_size, expected_means, ddof=0, axis=0)
    return (mean, stat.statistic, stat.pvalue)
control_clicks_chi = map(calc_chisquare_stats, control_clicks)
#%%
mean_plot = stats.probplot(map(lambda (m,s,p): m, control_clicks_chi), dist = "norm", plot = pylab)
#%%
stat_plot = stats.probplot(map(lambda (m,s,p): s, control_clicks_chi), dist = "chi2", sparams=(n_boot_samples-1), plot = pylab)
# В качестве ответа отметьте верные утвердения о значениях R^2, которые генерирует scipy.stats.probplot
# при отображении qq-графиков: одно c графика для среднего и одно с графика для выборочной суммы квадратов
# отклонения от выборочной суммы.
#%%
print "Mean R^2: %f" % mean_plot[1][2]
print "Chi square R^2: %f" % stat_plot[1][2]
# R2 для выборочного среднего получился больше, чем 0.99
# R2 для выборочной суммы квадратов отклонения от выборочной суммы получился больше, чем 0.99

# Одним из возможных аналогов t-критерия, которым можно воспрользоваться, является тест Манна-Уитни.
# На достаточно обширном классе распределений он является асимптотически более эффективным, чем t-критерий, и при этом не требует параметрических
# предположений о характере распределения.
# Разделите выборку на две части, соответствующие control и exp группам. Преобразуйте данные к виду, чтобы каждому пользователю соответствовало
# суммарное значение его кликов.
# С помощью критерия Манна-Уитни проверьте гипотезу о равенстве средних.
# Что можно сказать о получившемся значении достигаемого уровня значимости ? Выберите все правильные ответы
#%%
exp_data = frame[frame["slot"]=="exp"]
control_data = frame[frame["slot"]=="control"]

exp_user_clicks = exp_data.groupby("userID").agg("sum")["n_clicks"]
control_user_clicks = control_data.groupby("userID").agg("sum")["n_clicks"]

mann_whitney_stat = stats.mannwhitneyu(control_user_clicks, exp_user_clicks, alternative="two-sided")
print "Mann-Whitney criterion pvalue result: %.4f" % np.round(mann_whitney_stat.pvalue, 4)
mann_whitney_stat
# Получившееся значение достигаемого уровня значимости свидетельствует о статистической значимости отличий между двумя выборками

# Проверьте, для какого из браузеров наиболее сильно выражено отличие между количеством кликов в контрольной и
# экспериментальной группах.
# Для этого примените для каждого из срезов (по каждому из уникальных значений столбца browser) критерий Манна-Уитни
# между control и exp группами и сделайте поправку Холма-Бонферрони на множественную проверку с α=0.05.
# Какое заключение можно сделать исходя из полученных результатов?
# В качестве ответа введите количество незначимых изменений с точки зрения результатов, полученных после введения коррекции.
#%%
from statsmodels.sandbox.stats.multicomp import multipletests
exp_browser_names = set(exp_data["browser"])
exp_browser_data = {}
for browser_name in exp_browser_names:
    exp_browser_data[browser_name] = exp_data[exp_data["browser"]==browser_name]
control_browser_names = set(control_data["browser"])
control_browser_data = {}
for browser_name in control_browser_names:
    control_browser_data[browser_name] = control_data[control_data["browser"]==browser_name]
all_browser_names = exp_browser_names.union(control_browser_names)
mann_whitney_pvalues = []
for browser_name in all_browser_names:
    exp_b_user_clicks = exp_browser_data[browser_name]["n_clicks"].values
    control_b_user_clicks = control_browser_data[browser_name]["n_clicks"].values
    mann_whitney_stat = stats.mannwhitneyu(control_b_user_clicks, exp_b_user_clicks, alternative="two-sided")
    mann_whitney_pvalues.append(mann_whitney_stat.pvalue)
mann_whitney_pvalues_corrected = multipletests(mann_whitney_pvalues, alpha = 0.05, method = 'holm')
for b in zip(all_browser_names, mann_whitney_pvalues_corrected[0]):
    print "%s is statistically valuable: %r" % b
print "Count of not rejected hypothesis: %i" % len(filter(lambda x: not(x), mann_whitney_pvalues_corrected[0]))

# Для каждого браузера в каждой из двух групп (control и exp) посчитайте долю запросов, в которых пользователь не кликнул ни разу.
# Это можно сделать, поделив сумму значений n_nonclk_queries на сумму значений n_queries.
# Умножив это значение на 100, получим процент некликнутых запросов, который можно легче проинтерпретировать.
# Сходятся ли результаты проведенного Вами анализа с показателем процента некликнутых запросов?
# Отметьте все верные утверждения.
#%%
for browser_name in all_browser_names:
    exp_no_click_count = float(exp_browser_data[browser_name]["n_nonclk_queries"].sum())
    exp_queries_count = float(exp_browser_data[browser_name]["n_queries"].sum())
    control_no_click_count = float(control_browser_data[browser_name]["n_nonclk_queries"].sum())
    control_queries_count = float(control_browser_data[browser_name]["n_queries"].sum())
    exp_percent = exp_no_click_count/exp_queries_count*100
    control_percent = control_no_click_count/control_queries_count*100
    print "%s no click queries percentage: exp=%f\tcontrol=%f\tdiff=%f" % (browser_name, exp_percent, control_percent, (control_percent-exp_percent))