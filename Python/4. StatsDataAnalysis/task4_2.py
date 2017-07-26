# Instructions

# Для выполнения этого задания вам понадобятся данные о кредитных историях клиентов одного из банков.
# Поля в предоставляемых данных имеют следующий смысл:
#    LIMIT_BAL: размер кредитного лимита (в том числе и на семью клиента)
#    SEX: пол клиента (1 = мужской, 2 = женский )
#    EDUCATION: образование (0 = доктор, 1 = магистр; 2 = бакалавр; 3 = выпускник школы; 4 = начальное образование; 5= прочее; 6 = нет данных ).
#    MARRIAGE: (0 = отказываюсь отвечать; 1 = замужем/женат; 2 = холост; 3 = нет данных).
#    AGE: возраст в годах
#    PAY_0 - PAY_6 : История прошлых платежей по кредиту. PAY_6 - платеж в апреле, ... Pay_0 - платеж в сентябре.
#       Платеж = (0 = исправный платеж, 1=задержка в один месяц, 2=задержка в 2 месяца ...)
#    BILL_AMT1 - BILL_AMT6: задолженность, BILL_AMT6 - на апрель, BILL_AMT1 - на сентябрь
#    PAY_AMT1 - PAY_AMT6: сумма уплаченная в PAY_AMT6 - апреле, ..., PAY_AMT1 - сентябре
#    default - индикатор невозврата денежных средств
#%%
%pylab inline
import pandas as pd
import numpy as np

frame = pd.read_csv("..\..\Data\credit_card_default_analysis.csv", sep=",", header=0)
frame = pd.read_csv("credit_card_default_analysis.csv", sep=",", header=0)
frame.head()

# Задание
#    Размер кредитного лимита (LIMIT_BAL).
#       В двух группах, тех людей, кто вернул кредит (default = 0) и тех, кто его не вернул (default = 1) проверьте гипотезы:
#       a) о равенстве медианных значений кредитного лимита с помощью подходящей интервальной оценки
#%%
from scipy import stats
def get_bootstrap_samples(data, n_samples):
    data_length = len(data)
    indices = np.random.randint(0, data_length, (n_samples, data_length))
    return data[indices]
def stat_intervals(stat, alpha):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries
def calculate_median_confidence_interval(data, samples_count = 1000, alpha = 0.05):
    median = np.median(data)
    medians = map(lambda samples_group: np.median(samples_group), get_bootstrap_samples(data, samples_count))
    confint = stat_intervals(medians, alpha)
    return (median, confint)
def compare_distributions(data1, data2, samples_count, alpha):
    median_deltas = map(
        lambda (s_median,d_median): s_median-d_median,
        zip(
            map(lambda samples_group: np.median(samples_group), get_bootstrap_samples(data1, samples_count)),
            map(lambda samples_group: np.median(samples_group), get_bootstrap_samples(data2, samples_count))
        )
    )
    return stat_intervals(median_deltas, alpha)
#%%
# Разделим данные на две группы default и success
default_group = frame[frame["default"] == 1]
success_group = frame[frame["default"] == 0]
default_group_limits = default_group["LIMIT_BAL"].as_matrix()
success_group_limits = success_group["LIMIT_BAL"].as_matrix()
# Попытаемся визуально сравнить распределение кредитного лимита
frame.boxplot(column="LIMIT_BAL", by="default", figsize = (15,8))
pylab.show()
pylab.figure(figsize(15,5))
pylab.subplot(121)
pylab.xlabel("Success LIMIT_BAL")
pylab.hist(success_group_limits, label="Success limits")
pylab.subplot(122)
pylab.xlabel("Default LIMIT_BAL")
pylab.hist(default_group_limits, label="Default limits")
pylab.show()
# По графикам видно, что распределения лимитов для групп "success" и "default" очень похожи друг на друга, но есть разница
# между размерами кредитного лимита. Для группы success размер кредитного лимита выше. Докажем, с помощью доверительных
# интервалов, построенных с помощью метода bootstrap, что различие между значениями в этих группах действительно есть.
#%%
samples_count = 1000
alpha = 0.05
default_limit_confint = calculate_median_confidence_interval(default_group_limits, samples_count, alpha)
success_limit_confint = calculate_median_confidence_interval(success_group_limits, samples_count, alpha)
median_delta_confint = compare_distributions(success_group_limits, default_group_limits, samples_count, alpha)
print "Default limit max value: %.4f\tmin value: %.4f" % (np.max(default_group_limits), np.min(default_group_limits))
print "Success limit max value: %.4f\tmin value: %.4f" % (np.max(success_group_limits), np.min(success_group_limits))
print "Default median: %.4f\tConfidence interval: %s" % default_limit_confint
print "Success median: %.4f\tConfidence interval: %s" % success_limit_confint
print "Confidence interval for difference between \"default\" and \"success\" groups is %s" % median_delta_confint
print u"Доверительный интервал для разницы кредитного лимита между \"success\" и \"default\" группами далеко отстоит от нуля."
print u"Можно с уверенностью заявить, что кредитный лимит в группе \"success\" значимо выше."
#       b) о равенстве распределений с помощью одного из подходящих непараметрических критериев проверки равенства средних.
#       Значимы ли полученные результаты с практической точки зрения ?
#%%
# Мы имеем две независимые выборки "success" и "default". Однако мы не можем воспользоваться критерием Манна-Уитни,
# т.к. по графикам видно, что выборки имеют очень много повторяющихся значений. Поэтому будем использовать двухвыборочный
# перестановочный критерий для несвязанных выборок.
# Функции для подсчета перестановочного критерия
import itertools
def permutation_t_stat_ind(sample1, sample2):
    return np.median(sample1) - np.median(sample2)
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
def permutation_test(sample1, sample2, max_permutations = None, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")
    
    t_stat = permutation_t_stat_ind(sample1, sample2)
    
    zero_distr = permutation_zero_dist_ind(sample1, sample2, max_permutations)
    
    if alternative == 'two-sided':
        return sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)
    
    if alternative == 'less':
        return sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        return sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)
#%%
print "Permutation criterion p-value: %.4f" % permutation_test(default_group_limits, success_group_limits, max_permutations=1000, alternative="two-sided")

# Доверительные интервалы для медиан в группах "success" и "default" далеко отстоят друг от друга.
# Доверительный интервал для разницы медианных значений в этих группах далеко отстоит от нуля.
# Непараметрический перестановочный критерий для двух независимых выборок дает достигаемый уровень значимости
# сильно меньше 0.05.
# Из этого можно сделать вывод, что гипотеза о равенстве распределений отвергается.
# С практической точки зрения эти результаты говорят о том, что банк выдает большие кредиты надежным клиентам не
# просрачивающим выплаты.

#    Пол (SEX):
#       Проверьте гипотезу о том, что гендерный состав группы людей вернувших и не вернувших кредит отличается.
#       Хорошо, если вы предоставите несколько различных решений этой задачи (с помощью доверительного интервала и подходящего статистического критерия)
#%%
default_group_sex = default_group["SEX"].as_matrix()
success_group_sex = success_group["SEX"].as_matrix()
#%%
def proportions_confint_diff(first_vector, second_vector, alpha = 0.05):
    count1 = float(len(first_vector))
    count2 = float(len(second_vector))
    p1 = float(len(filter(lambda x: x == 1, first_vector)))/count1
    p2 = float(len(filter(lambda x: x == 1, second_vector)))/count2
    z = stats.norm.ppf(1. - alpha / 2.)

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1. - p1)/ count1 + p2 * (1 - p2)/ count2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1. - p1)/ count1 + p2 * (1 - p2)/ count2)
    
    return (left_boundary, right_boundary)

def proportions_diff_z_stat_ind(sample1, sample2):
    n1 = len(sample1)
    n2 = len(sample2)
    
    p1 = float(len(filter(lambda x: x == 1, sample1))) / n1
    p2 = float(len(filter(lambda x: x == 1, sample2))) / n2 
    P = float(p1*n1 + p2*n2) / (n1 + n2)
    
    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))

def proportions_diff_z_test(sample1, sample2, alternative = 'two-sided'):
    z_stat = proportions_diff_z_stat_ind(sample1, sample2)
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")
    
    if alternative == 'two-sided':
        return 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    
    if alternative == 'less':
        return stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - stats.norm.cdf(z_stat)

# Функция, которая строит bar chart для таблицы сопряженности
def build_contingency_table_bar_chart(contingency_table, row_groups, column_groups):
    for (row_index,row_name) in enumerate(row_groups):
        column_counts = map(lambda cell: float(cell), contingency_table[row_index])
        overall_count = sum(column_counts)
        if overall_count > 0:
            for (column_count,column_name) in zip(column_counts, column_groups):
                pylab.bar(row_index, column_count/overall_count, label = ("%s %s (%.0f)" % (row_name, column_name, column_count)))
    pylab.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pylab.show()
#%%
sex_contingency_table = pd.pivot_table(
    frame[["SEX", "default", "LIMIT_BAL"]],
    values=["LIMIT_BAL"],
    index=["SEX"],
    columns=["default"],
    fill_value = 0,
    aggfunc='count')
sex_contingency_matrix = sex_contingency_table.as_matrix()
sex_contingency_table
#%%
build_contingency_table_bar_chart(sex_contingency_matrix, ["Men", "Women"], ["success", "default"])

# Визуально кажется, что доля мужчин вернувших кредит (Men success) меньше доли женщин вернувших кредит (Women success)
# Попробуем подтвердить это с помощью доверительных интервалов.
# Сравним доли мужчин в группах "default" и "success" на основе доверительных интервалов
#%%
print "Proportion confidential interval is [%.4f; %.4f]" % proportions_confint_diff(default_group_sex, success_group_sex)
# Доверительный интервал не содержит 0 и находится от него справа, соответственно можно утверждать,
# что доля мужчин не вернувших кредит больше, чем доля вернувших.
# Теперь сравним доли мужчин в тех-же группах на основе z-критерия для двух независимых выборок
pvalue = proportions_diff_z_test(default_group_sex, success_group_sex, alternative='two-sided')
print "p-value: ", pvalue
# Достигаемый уровень значимости Z-критерия p-value для этих двух выборок сильно меньше 0.05.
# Гипотеза о равенстве доли мужчин в выборках "default" и "success" с уверенностью отвергается.
# По значениям доверительного итервала можно сказать, что мужчины возвращают кредит реже, чем женщины.
# Доверительный интервал находится очень близко к нулю. На мой взгляд это значит, что с практической точки зрения разница
# в гендерном составе не является значимой.

#    Образование (EDUCATION):
#       Проверьте гипотезу о том, что образование не влияет на то, вернет ли человек долг.
#       Предложите способ наглядного представления разницы в ожидаемых и наблюдаемых значениях количества человек вернувших и не вернувших долг.
#       Например, составьте таблицу сопряженности "образование" на "возврат долга", где значением ячейки была бы разность между наблюдаемым и ожидаемым количеством
#           человек.
#       Как бы вы предложили модифицировать таблицу так, чтобы привести значения ячеек к одному масштабу не потеряв в интерпретируемости?
#       Наличие какого образования является наилучшим индикатором того, что человек отдаст долг?
#       Наоборот, не отдаст долг?
#%%
success_group_edu = success_group["EDUCATION"].as_matrix()
default_group_edu = default_group["EDUCATION"].as_matrix()
pylab.hist([success_group_edu, default_group_edu], 7, label = ["Success", "Default"])
pylab.legend()
# По такому графику сложно как-то интерпретировать разницу в ожидаемых и наблюдаемых значениях количества человек вернувших
# и не вернувших долг. Очевидно, разных группах обучения разное количество людей. Выборки не масштабированы
#%%
def v_Cramer_correlation(table):
    chi_stat = stats.chi2_contingency(table)[0]
    k_min = np.min(table.shape)
    n = np.sum(table)
    return np.sqrt(chi_stat/(n*(k_min-1)))
#%%
# Построим таблицу сопряженности, в ячейках этой таблицы находится количество человек, для которых значения признаков совпадают
edu_contingency_table = pd.pivot_table(
    frame[["EDUCATION", "default", "LIMIT_BAL"]],
    values=["LIMIT_BAL"],
    index=["EDUCATION"],
    columns=["default"],
    fill_value = 0,
    aggfunc='count')
edu_contingency_matrix = edu_contingency_table.as_matrix()
#%%
# Для этой таблицы сопряженности построим bar chart, где сравниваются отношения значений success/count и default/count.
# Эти значения фактически представляют собой долю клиентов вернувших кредит и долю клиентов просрочивших выплаты.
# Эти доли принимают значения от 0 до 1 и соответственно имеют одиаковый масштаб.
edu_contingency_table
#%%
build_contingency_table_bar_chart(edu_contingency_matrix, ["doctor","master","bachelor","scholar","basic","other","n/a"], ["success", "default"])
# Поскольку оба признака "EDUCATION" и "default" де-факто категориальные, мы можем посчитать меру взаимосвязи между ними
# с помощью коэффициента V-Крамера.
# Достигаемый уровень значимости для этой статистики можно посчитать с помощью критерия хи-квадрат.
# Проверим корректность использования этого критерия:
#%%
edu_contingency_matrix_cells = edu_contingency_matrix.reshape(-1)
count_is_enough = len(filter(lambda val: val > 5, edu_contingency_matrix_cells))
all_cells_count = len(edu_contingency_matrix_cells)
elements_count = sum(edu_contingency_matrix_cells)
print u"Общее количество элементов: %i. Процент ячеек, где количество элементов больше 5: %.4f%%" % (elements_count,(float(count_is_enough)*100/all_cells_count))
# Число ячеек, где количество элементов больше 5 составляет более 80%, а количество элементов сильно больше 40.
# Применение критерия хи-квадрат корректно.
#%%
print "V-Cramer statistic is: %.4f" % v_Cramer_correlation(edu_contingency_matrix)
print "p-value:", stats.chi2_contingency(edu_contingency_matrix)[1]
# Значение коэффициента V Крамера не очень велико, однако p-value критерия хи-квадрат говорит нам, что это значение
# статистически значимо.
# Судя по графику чаще всего отдают долг люди с докторской степенью.
# Реже всего, - люди со школьным образованием.
# Статистика V-Крамера достаточно мала, так что нельзя использовать этот признак как главный.

#    Семейное положение (MARRIAGE):
#       Проверьте, как связан семейный статус с индикатором дефолта:
#           нужно предложить меру, по которой можно измерить возможную связь этих переменных и посчитать ее значение.
#%%
marriage_contingency_table = pd.pivot_table(
    frame[["MARRIAGE", "default", "LIMIT_BAL"]],
    values=["LIMIT_BAL"],
    index=["MARRIAGE"],
    columns=["default"],
    fill_value = 0,
    aggfunc='count')
marriage_contingency_matrix = marriage_contingency_table.as_matrix()
marriage_contingency_table
# Для этой таблицы сопряженности построим bar chart, где сравниваются отношения значений success/count и default/count.
# Эти значения фактически представляют собой долю клиентов вернувших кредит и долю клиентов просрочивших выплаты.
# Эти доли принимают значения от 0 до 1 и соответственно имеют одиаковый масштаб.
#%%
build_contingency_table_bar_chart(marriage_contingency_matrix, ["never tell","married","free","n/a"], ["success", "default"])
#%%
# Поскольку оба признака "MARRIAGE" и "default" де-факто категориальные, мы можем посчитать меру взаимосвязи между ними с
# помощью коэффициента V-Крамера.
# Достигаемый уровень значимости для этой статистики можно посчитать с помощью критерия хи-квадрат.
# Проверим корректность использования этого критерия:
marriage_contingency_matrix_cells = marriage_contingency_matrix.reshape(-1)
mar_count_is_enough = len(filter(lambda val: val > 5, marriage_contingency_matrix_cells))
mar_all_cells_count = len(marriage_contingency_matrix_cells)
mar_elements_count = sum(marriage_contingency_matrix_cells)
print u"Общее количество элементов: %i. Процент ячеек, где количество элементов больше 5: %.4f%%" % (mar_elements_count,(float(mar_count_is_enough)*100/mar_all_cells_count))
# Число ячеек, где количество элементов больше 5 составляет более 80%, а количество элементов сильно больше 40.
# Применение критерия хи-квадрат корректно.
#%%
print "V-Cramer statistic is: %.4f" % v_Cramer_correlation(marriage_contingency_matrix)
print "p-value:", stats.chi2_contingency(marriage_contingency_matrix)[1]
# Значение коэффициента V Крамера не очень велико, однако p-value критерия хи-квадрат говорит нам, что это значение
# статистически значимо.
# Значение признака MARRIAGE влияет на возврат долга, но несущественно.

#    Возраст (AGE):
#       Относительно двух групп людей вернувших и не вернувших кредит проверьте следующие гипотезы:
#%%
default_group_ages = default_group["AGE"].as_matrix()
success_group_ages = success_group["AGE"].as_matrix()
frame.boxplot(column="AGE", by="default", figsize = (15,8))
pylab.show()
pylab.figure(figsize(15,5))
pylab.subplot(121)
pylab.xlabel("Success ages")
pylab.hist(success_group_ages, label="Success ages")
pylab.subplot(122)
pylab.xlabel("Default ages")
pylab.hist(default_group_ages, label="Default ages")
pylab.show()
# По графикам видно, что медианные значения возрастов практически не отличаются. Проверим так ли это.
#       a) о равенстве медианных значений возрастов людей
#%%
default_ages_confint = calculate_median_confidence_interval(default_group_ages, samples_count, alpha)
success_ages_confint = calculate_median_confidence_interval(success_group_ages, samples_count, alpha)
median_delta_ages_confint = compare_distributions(success_group_ages, default_group_ages, samples_count, alpha)
print "Default ages max value: %.4f\tmin value: %.4f" % (np.max(default_group_ages), np.min(default_group_ages))
print "Success ages max value: %.4f\tmin value: %.4f" % (np.max(success_group_ages), np.min(success_group_ages))
print "Default ages median: %.4f\tConfidence interval: %s" % default_ages_confint
print "Success ages median: %.4f\tConfidence interval: %s" % success_ages_confint
print "Confidence interval for difference between ages in \"default\" and \"success\" groups is %s" % median_delta_ages_confint
print u"Доверительный интервал для разницы возрастов между \"success\" и \"default\" группами содержит ноль."
print u"Гипотеза о равенстве возрастов в этих группах не отвергается."
#       b) о равенстве распределений с помощью одного из подходящих непараметрических критериев проверки равенства средних.
# По гистограммам можно определить, что в выборках много повторяющихся значений. Использование критерия Манна-Уитни
# некорректно, поэтому используем перестановочный критерий для двух независимых выборок
#%%
print "Permutation criterion p-value: %.4f" % permutation_test(default_group_ages, success_group_ages, max_permutations=1000, alternative="two-sided")

# Гипотеза о равенстве медианных значений возрастов для вернувших и просрочивших кредит перестановочным критерием не
# отвергается.

# Значимы ли полученные результаты с практической точки зрения ?

## Заключение
# Мы определили, что
# 
# * размер кредитного лимита для людей не допускающих просрочки значимо выше.
# * гендерный состав людей вернувших и просрочивших долг значимо различается.
# * доля женщин в группе не допускающих просрочку выше.
# * образование не сильно, но все-таки значимо влияет на вероятность возврата долга.
# * можно утверждать, что люди с лучшим образованием чаще возвращают долг.
#   (хотя здесь статистику портят группы other, n/a и basic).
# * незамужние люди и люди не желающие раскрывать свой семейный статус чаще возвращают долг.
#   Сомнительное утверждение, требующее дополнительной проверки, тем более, что статистика V-Крамера очень мала.
# * возрастной состав в группах вернувших и просрочивших долг примерно одинаков.
# 
# На мой взгляд полученные результаты совпадают с очень частой политикой выдачи кредитов банками.
# С другой стороны стоит отметить, что эти признаки скорее выглядят вспомогательными и первоочередными критериями скорее
# являются доход и трудоустроенность клиента.
