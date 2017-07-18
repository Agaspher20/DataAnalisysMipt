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
frame.head()

# Задание
#    Размер кредитного лимита (LIMIT_BAL).
#       В двух группах, тех людей, кто вернул кредит (default = 0) и тех, кто его не вернул (default = 1) проверьте гипотезы:
#       a) о равенстве медианных значений кредитного лимита с помощью подходящей интервальной оценки
#%%
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
#%%
default_group = frame[frame["default"] == 1]
success_group = frame[frame["default"] == 0]
default_group_limits = default_group["LIMIT_BAL"].as_matrix()
success_group_limits = success_group["LIMIT_BAL"].as_matrix()
#%%
pylab.ylabel("LIMIT_BAL")
pylab.plot(sorted(list(set(success_group_limits))), label="Success limits")
pylab.plot(sorted(list(set(default_group_limits))), label="Default limits")
pylab.legend()
pylab.show()
#%%
samples_count = 1000
alpha = 0.05
default_limit_confint = calculate_median_confidence_interval(default_group_limits, samples_count, alpha)
success_limit_confint = calculate_median_confidence_interval(success_group["LIMIT_BAL"].as_matrix(), samples_count, alpha)
median_deltas = map(
    lambda (s_median,d_median): s_median-d_median,
    zip(
        map(lambda samples_group: np.median(samples_group), get_bootstrap_samples(success_group["LIMIT_BAL"].as_matrix(), samples_count)),
        map(lambda samples_group: np.median(samples_group), get_bootstrap_samples(default_group["LIMIT_BAL"].as_matrix(), samples_count))
    )
)
median_delta_confint = stat_intervals(median_deltas, alpha)
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
from scipy import stats
print "Mann-Whitney criterion p-value:"
print stats.mannwhitneyu(default_group_limits, success_group_limits, alternative="two-sided").pvalue
print "Wilcoxon criterion p-value:"
print stats.wilcoxon(
    np.array(sorted(success_group_limits)[0:len(default_group_limits)])-np.array(sorted(default_group_limits))).pvalue

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
#%%
print "Proportion confidential interval is [%.4f; %.4f]" % proportions_confint_diff(default_group_sex, success_group_sex)

pvalue = proportions_diff_z_test(default_group_sex, success_group_sex, alternative='two-sided')
print "p-value: ", pvalue

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
pylab.show()
for edu_type in [(0,"doctor"),(1,"master"),(2,"bachelor"),(3,"scholar"),(4,"basic"),(5,"other"),(6,"n/a")]:
    success_edu_count = float(len(filter(lambda val: val == edu_type[0], success_group_edu)))
    default_edu_count = float(len(filter(lambda val: val == edu_type[0], default_group_edu)))
    overall_count = success_edu_count+default_edu_count
    if overall_count > 0:    
        pylab.bar(edu_type[0], success_edu_count/overall_count, label = (edu_type[1] + (" success (%.0f)" % success_edu_count)))
        pylab.bar(edu_type[0], default_edu_count/overall_count, label = (edu_type[1] + (" default (%.0f)" % default_edu_count)))
pylab.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
pylab.show()
#%%
#    Семейное положение (MARRIAGE):
#       Проверьте, как связан семейный статус с индикатором дефолта:
#           нужно предложить меру, по которой можно измерить возможную связь этих переменных и посчитать ее значение.
#    Возраст (AGE):
#       Относительно двух групп людей вернувших и не вернувших кредит проверьте следующие гипотезы:
#       a) о равенстве медианных значений возрастов людей
#       b) о равенстве распределений с помощью одного из подходящих непараметрических критериев проверки равенства средних.
#       Значимы ли полученные результаты с практической точки зрения ?

#Review criteria
#    Выполнение каждого пункта задания должно начинаться с графика с данными, которые вы собираетесь анализировать.
#       Еще лучше, если вы разложите графики анализируемого фактора по переменной (default), на которую хотите изучить влияние этого фактора,
#       и проинтерпретируете отличия в полученных распределениях.
#    При использовании статистических критериев необходимо убедиться в том, что условия их применимости выполняются.
#       Например, если вы видите, что данные бинарные, то не нужно применять критерий Стьюдента.
#    При каждом использовании любого критерия необходимо указать, какая проверяется гипотеза, против какой альтернативы, чему равен достигаемый уровень значимости,
#       принимается или отвергается нулевая гипотеза на уровне значимости 0.05. Если задача позволяет, нужно оценить размер эффекта и предположить, имеет ли этот
#       результат практическую значимость.
#    Выполненное задание необходимо представить в ipython-ноутбуке.
