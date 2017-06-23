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
#%%
def stat_intervals(stat, alpha):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries
#%%
exp_clicks = get_bootstrap_samples(frame[frame["slot"]=="exp"]["n_clicks"].values, 1000)
control_clicks = get_bootstrap_samples(frame[frame["slot"]=="control"]["n_clicks"].values, 1000)

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