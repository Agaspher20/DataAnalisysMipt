# Для 61 большого города в Англии и Уэльсе известны средняя годовая смертность на 100000 населения (по данным 1958–1964)
# и концентрация кальция в питьевой воде (в частях на маллион).
# Чем выше концентрация кальция, тем жёстче вода. Города дополнительно поделены на северные и южные.
#%%
import pandas as pd
import numpy as np
frame = pd.read_csv("..\..\Data\water.txt", sep="\t", header=0)
frame.head()
# Есть ли связь между жёсткостью воды и средней годовой смертностью? Посчитайте значение коэффициента корреляции Пирсона между этими признаками,
# округлите его до четырёх знаков после десятичной точки.\
#%%
print "Pearson correlation mortality-hardness: %.4f" % np.round(frame.corr(method="pearson")["mortality"]["hardness"], 4)

# В предыдущей задаче посчитайте значение коэффициента корреляции Спирмена между средней годовой смертностью и жёсткостью воды.
# Округлите до четырёх знаков после десятичной точки.
#%%
print "Spearman correlation mortality-hardness: %.4f" % np.round(frame.corr(method="spearman")["mortality"]["hardness"], 4)

# Сохраняется ли связь между признаками, если разбить выборку на северные и южные города?
# Посчитайте значения корреляции Пирсона между средней годовой смертностью и жёсткостью воды в каждой из двух подвыборок,
# введите наименьшее по модулю из двух значений, округлив его до четырёх знаков после десятичной точки.
#%%
south_frame = frame[frame["location"] == "South"]
north_frame = frame[frame["location"] == "North"]
south_corr = south_frame.corr(method="pearson")["mortality"]["hardness"]
north_corr = north_frame.corr(method="pearson")["mortality"]["hardness"]

if(np.abs(south_corr) < np.abs(north_corr)):
    print "Pearson correlation for south cities is lower: %.4f" % np.round(south_corr, 4)
else:
    print "Pearson correlation for north cities is lower: %.4f" % np.round(north_corr, 4)

# Среди респондентов General Social Survey 2014 года хотя бы раз в месяц проводят вечер в баре 203 женщины и 239 мужчин;
# реже, чем раз в месяц, это делают 718 женщин и 515 мужчин.
# Посчитайте значение коэффициента корреляции Мэтьюса между полом и частотой похода в бары.
# Округлите значение до трёх знаков после десятичной точки.
#%%
def Matthews_correlation(a, b, c, d):
    return (a*d-b*c)/np.sqrt((a+b)*(a+c)*(b+d)*(c+d))
#%%
men_visit_monthly = 239
women_visit_monthly = 203
men_visit_rarely = 515
women_visit_rarely = 718
print "Matthews correlation is: %.3f" % np.round(Matthews_correlation(men_visit_monthly, men_visit_rarely, women_visit_monthly, women_visit_rarely), 3)

# В предыдущей задаче проверьте, значимо ли коэффициент корреляции Мэтьюса отличается от нуля.
# Посчитайте достигаемый уровень значимости; используйте функцию scipy.stats.chi2_contingency.
# Введите номер первой значащей цифры (например, если вы получили 5.5×10−8, нужно ввести 8).
#%%
from scipy.stats import chi2_contingency
chi_stat = chi2_contingency([[men_visit_monthly, men_visit_rarely], [women_visit_monthly, women_visit_rarely]])
print "Chi2 p-value is: %f" % chi_stat[1]

# В предыдущей задаче давайте попробуем ответить на немного другой вопрос: отличаются ли доля мужчин и доля женщин, относительно часто проводящих вечера в баре?
# Постройте 95% доверительный интервал для разности долей, вычитая долю женщин из доли мужчин.
# Чему равна его нижняя граница? Округлите до четырёх знаков после десятичной точки.
#%%
from scipy.stats import norm
def proportions_confint_diff_ind(p1, count1, p2, count2, alpha = 0.05):    
    z = norm.ppf(1 - alpha / 2.)
    
    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1. - p1)/ count1 + p2 * (1 - p2)/ count2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1. - p1)/ count1 + p2 * (1 - p2)/ count2)
    
    return (left_boundary, right_boundary)
#%%
all_men = men_visit_monthly + men_visit_rarely
all_women = women_visit_monthly + women_visit_rarely
men_proportion = float(men_visit_monthly)/float(all_men)
women_proportion = float(women_visit_monthly)/float(all_women)
conf_interval_diff = proportions_confint_diff_ind(men_proportion, all_men, women_proportion, all_women)
print "Interval for men and women bar visitors proportion diff: [%.4f, %.4f]" % (np.round(conf_interval_diff[0],4),np.round(conf_interval_diff[1],4))

# Проверьте гипотезу о равенстве долей любителей часто проводить вечера в баре среди мужчин и женщин.
# Посчитайте достигаемый уровень значимости, используя двустороннюю альтернативу.
# Введите номер первой значащей цифры (например, если вы получили 5.5×10−8, нужно ввести 8).
#%%
def proportions_diff_z_stat_ind(p1, count1, p2, count2, alpha = 0.05):
    P = float(p1*count1 + p2*count2) / (count1 + count2)
    
    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / count1 + 1. / count2))

def proportions_diff_z_test(z_stat, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")
    
    if alternative == 'two-sided':
        return 2 * (1 - norm.cdf(np.abs(z_stat)))
    
    if alternative == 'less':
        return norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - norm.cdf(z_stat)

#%%
pvalue = proportions_diff_z_test(proportions_diff_z_stat_ind(men_proportion, all_men, women_proportion, all_women))
print "p-value for men-women proportion difference is: %f" % pvalue

# Посмотрим на данные General Social Survey 2014 года и проанализируем, как связаны ответы на вопросы "Счастливы ли вы?"
# и "Довольны ли вы вашим финансовым положением?"
#                     Не доволен    Более или менее    Доволен
# Не очень счастлив	         197                111         33
# Достаточно счастлив        382                685        331
# Очень счастлив             110                342        333
# Чему равно значение статистики хи-квадрат для этой таблицы сопряжённости? Округлите ответ до четырёх знаков после десятичной точки.
#%%
contingency_table = np.array([[197, 111, 33], [382, 685, 331], [110, 342, 333]])
chi_stat_happiness = chi2_contingency(contingency_table)
print "Chi2 statistic value is: %.4f" % np.round(chi_stat_happiness[0], 4)

# На данных из предыдущего вопроса посчитайте значение достигаемого уровня значимости.
# Введите номер первой значащей цифры (например, если вы получили 5.5×10−8, нужно ввести 8).
#%%
print "Chi2 p-value is: "
print chi_stat_happiness[1]

# Чему в предыдущей задаче равно значение коэффициента V Крамера для рассматриваемых признаков? Округлите ответ до четырёх знаков после десятичной точки. 
#%%
def v_Cramer_correlation(table):
    chi_stat = chi2_contingency(table)[0]
    k_min = np.min(table)
    n = np.sum(table)
    return np.sqrt(chi_stat/(n*k_min))
#%%
print "V-Cramer statistic is: %.4f" % np.round(v_Cramer_correlation(contingency_table), 4)
