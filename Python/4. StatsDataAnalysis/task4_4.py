
# В этом задании вам предлагается проанализировать данные одной из американских телекоммуникационных компаний о
# пользователях, которые потенциально могут уйти.
#%%
%pylab inline
import pandas as pd
import numpy as np

frame = pd.read_csv("..\..\Data\churn_analysis.csv", sep=",", header=0)
frame.head()
# Измерены следующие признаки:
#    state — штат США
#    account_length — длительность использования аккаунта
#    area_code — деление пользователей на псевдорегионы, использующееся в телекоме
#    intl_plan — подключена ли у пользователя услуга международного общения
#    vmail_plan — подключена ли у пользователя услуга голосовых сообщений
#    vmail_message — количество голосых сообщений, который пользователь отправил / принял
#    day_calls — сколько пользователь совершил дневных звонков
#    day_mins — сколько пользователь проговорил минут в течение дня
#    day_charge — сколько пользователь заплатил за свою дневную активность
#    eve_calls, eve_mins, eve_charge — аналогичные метрики относительно вечерней активности
#    night_calls, night_mins, night_charge — аналогичные метрики относительно ночной активности
#    intl_calls, intl_mins, intl_charge — аналогичные метрики относительно международного общения
#    custserv_calls — сколько раз пользователь позвонил в службу поддержки
#    treatment — номер стратегии, которая применялись для удержания абонентов (0, 2 = два разных типа воздействия, 1 = контрольная группа)
#    mes_estim — оценка интенсивности пользования интернет мессенджерами
#    churn — результат оттока: перестал ли абонент пользоваться услугами оператора
# Давайте рассмотрим всех пользователей из контрольной группы (treatment = 1). Для таких пользователей мы хотим
# проверить гипотезу о том, что штат абонента не влияет на то, перестанет ли абонент пользоваться услугами оператора.
# Для этого мы воспользуемся критерием хи-квадрат.
#  Постройте таблицы сопряженности между каждой из всех 1275 возможных неупорядоченных пар штатов и значением признака churn.
#  Для каждой такой таблицы 2x2 применить критерий хи-квадрат можно с помощью функции
#    scipy.stats.chi2_contingency(subtable, correction=False)
# Заметьте, что, например, (AZ, HI) и (HI, AZ) — это одна и та же пара.
# Обязательно выставьте correction=False (о том, что это значит, вы узнаете из следующих вопросов).
# Сколько достигаемых уровней значимости оказались меньше, чем α=0.05?
#%%
control_group = frame[frame["treatment"] == 1]
states = list(set(control_group["state"].values))
control_states_pivot = pd.pivot_table(
    control_group,
    values=["treatment"],
    index=["state"],
    columns=["churn"],
    fill_value = 0,
    aggfunc='count')
#%%
not_enough_data_count = len(filter(lambda val: val < 5, control_states_pivot.loc[:,[False, True]].values))
not_enough_data_count += len(filter(lambda val: val < 5, control_states_pivot.loc[:,[True, False]].values))
print "Count of cells where is not enough data is %i. The percent of these cells is %.2f%%" % (not_enough_data_count, float(not_enough_data_count)/(control_states_pivot.shape[0]*control_states_pivot.shape[1])*100)
#%%
from scipy import stats
def calculate_pairwise_diffs(pivot_table, states, stat_calculator):
    states_count = len(states)
    result = []
    for i in xrange(states_count-1):
        first_state = states[i]
        for j in xrange(i+1, states_count):
            second_state = states[j]
            chi_2_stat = stat_calculator(pivot_table.loc[[first_state,second_state],:])
            result.append(chi_2_stat)
    return result
def stats_compare_plot(control_stat, validate_stat):
    sorted_stats,sorted_stats_corrected = zip(*sorted(
        map(
            lambda (stat, c_stat): (stat[1],c_stat[1]),
            zip(control_stat,validate_stat)),
        key=(lambda (stat, c_stat): stat)))
    sorted_stats_x,sorted_stats_y=zip(*enumerate(sorted_stats))
    sorted_stats_corrected_x,sorted_stats_corrected_y=zip(*enumerate(sorted_stats_corrected))
    pylab.xlim(0,len(sorted_stats))
    pylab.scatter(sorted_stats_x, sorted_stats_y, color="r", alpha=.5)
    pylab.scatter(sorted_stats_corrected_x, sorted_stats_corrected_y, color="g", alpha=.5)
    pylab.show()
#%%
def chi_2_stat_no_correction(table):
    return stats.chi2_contingency(table, correction=False)
chi_2_stats = calculate_pairwise_diffs(control_states_pivot, states, chi_2_stat_no_correction)
print "A number of cells where p-value is less than 0.05 for chi-square criterion: %i" % len(filter(lambda stat: stat[1] < 0.05, chi_2_stats))
# Какие проблемы Вы видите в построении анализа из первого вопроса? Отметьте все верные утверждения. 
#    Интерпретация числа достигаемых уровней значимости, меньших α=0.05, некорректна, поскольку не сделана поправка
#      на множественную проверку гипотез.
#    Применение критерия xи-квадрат для этих данных не обосновано, потому что не выполняются условия, при которых этот
#      критерий дает правильные результаты.


# В основе критерия xи-квадрат лежит предположение о том, что если верна нулевая гипотеза, то дискретное биномиальное
# распределение данных по клеткам в таблице сопряженности может быть аппроксимировано с помощью непрерывного распределения
# xи-квадрат. Однако точность такой аппроксимации существенно зависит от суммарного количества наблюдений и их
# распределения в этой таблице (отсюда и ограничения при использовании критерия xи-квадрат).
# Одним из способов коррекции точности аппроксимации является поправка Йетса на непрерывность. Эта поправка заключается
# в вычитании константы 0.5 из каждого модуля разности наблюденного Oi и ожидаемого Ei значений, то есть, статистика
# с такой поправкой выглядит так: 
#    χ^2Yates=∑{i=1->N}(|Oi−Ei|−0.5)^2/Ei
# Такая поправка, как несложно догадаться по формуле, как правило, уменьшает значение статистики χ^2, то есть
# увеличивает достигаемый уровень значимости.
# Эта поправка обычно используется для таблиц сопряженности размером 2x2 и для небольшого количества наблюдений.
# Такая поправка, однако, не является серебрянной пулей, и часто критикуется за то, что статистический критерий при
# ее использовании становится слишком консервативным, то есть часто не отвергает нулевую гипотезу там, где она
# неверна (совершает ошибку II рода).
# Полезно знать, что эта поправка часто включена по умолчанию (например, в функции scipy.stats.chi2_contingency) и
# понимать ее влияние на оценку достигаемого уровня значимости.
# Проведите те же самые сравнения, что и в вопросе №1, только с включенной коррекцией
#    scipy.stats.chi2_contingency(subtable, correction=True)
# и сравните полученные результаты, отметив все верные варианты.
#%%
def chi_2_stat_correction(table):
    return stats.chi2_contingency(table, correction=True)
chi_2_corrected_stats = calculate_pairwise_diffs(control_states_pivot, states, chi_2_stat_correction)
print "A number of cells where p-value is less than 0.05 for chi-square criterion with correction: %i" % len(filter(lambda stat: stat[1] < 0.05, chi_2_corrected_stats))
#    Количество достигаемых уровней значимости, меньших, чем 0.05, в точности равно нулю. То есть поправка увеличила
#      достигаемые уровни значимости настолько, что больше ни одно из значений достигаемого уровня значимости не
#      попадает в диапазон от 0 до 0.05.
corrected_stats_greater_count = len(filter(lambda(stat, corrected): corrected[1] > stat[1], zip(chi_2_stats, chi_2_corrected_stats)))
corrected_stats_less_count = len(filter(lambda(stat, corrected): corrected[1] < stat[1], zip(chi_2_stats, chi_2_corrected_stats)))
stats_compare_plot(chi_2_stats, chi_2_corrected_stats)
print "A number of corrected stats where p-value is greater than p-value without correction is %i. Percentage: %.2f" % (corrected_stats_greater_count, float(corrected_stats_greater_count)/len(chi_2_stats)*100)
print "A number of corrected stats where p-value is less than p-value without correction is %i. Percentage: %.2f" % (corrected_stats_less_count, float(corrected_stats_less_count)/len(chi_2_stats)*100)
#    Достигаемые уровни значимости на наших данных, полученные с помощью критерия xи-квадрат с поправкой Йетса, в
#      среднем получаются больше, чем соответствующие значения без поправки.


# Что если у нас мало данных, мы не хотим использовать аппроксимацию дискретного распределения непрерывным и
# использовать сомнительную поправку, предположения критерия xи-квадрат не выполняются, а проверить гипотезу о том,
# что данные принадлежат одному распределению, нужно?
# В таком случае прибегают к так называемому точному критерию Фишера. Этот критерий не использует приближений и в
# точности вычисляет значение достигаемого уровня значимости используя комбинаторный подход.
# Пусть у нас есть таблица сопряженности 2x2:
#                  Группа 1    Группа 2    Σ
# Воздействие 1           a           b    a+b
# Воздействие 2           c           d    c+d
#             Σ         a+c         b+d    n=a+b+c+d
# Тогда вероятность получить именно такие a,b,c,d при фиксированных значениях сумм по строкам и по столбцам) задается
# выражением
#    p=(a+b)!(c+d)!(a+c)!(b+d)!/(a! b! c! d! n!)
# В числителе этой дроби стоит суммарное количество способов выбрать a и c из a+b и c+d соответственно.
# А в знаменателе — количество способов выбрать число объектов, равное сумме элементов первого столбца a+c из общего
# количества рассматриваемых объектов n.
# Чтобы посчитать достигаемый уровень значимости критерия Фишера, нужно перебрать все возможные значения a,b,c,d, в
# клетках этой таблицы так, чтобы построковые и постолбцовые суммы не изменились. Для каждого такого набора a,b,c,d 
# нужно вычислить значение pi по формуле выше и просуммировать все такие значения pi, которые меньше или равны p,
# которое мы вычислили по наблюдаемым значениям a,b,c,d.
# Понятно, что такой критерий вычислительно неудобен в силу большого количества факториалов в формуле выше. То есть
# даже при небольших выборках для вычисления значения этого критерия приходится оперировать очень большими числами.
# Поэтому данным критерием пользуются обычно только для таблиц 2x2, но сам критерий никак не ограничен количеством
# строк и столбцов, и его можно построить для любой таблицы n×m.
# Посчитайте для каждой пары штатов, как и в первом задании, достигаемый уровень значимости с помощью точного критерия
# Фишера и сравните получившиеся значения с двумя другими подходами, описанными выше.
# Точный критерий Фишера удобно вычислять с помощью функции
#     scipy.stats.fisher_exact
# которая принимает на вход таблицу сопряженности 2x2.
#%%
def fisher_exact_stat(table):
    return stats.fisher_exact(table, alternative="two-sided")
fischer_stats = calculate_pairwise_diffs(control_states_pivot, states, fisher_exact_stat)
fischer_stats_greater_count_c = len(filter(lambda(corrected, fischer): fischer[1] > corrected[1], zip(chi_2_corrected_stats, fischer_stats)))
fischer_stats_less_count_c = len(filter(lambda(corrected, fischer): fischer[1] < corrected[1], zip(chi_2_corrected_stats, fischer_stats)))
fischer_stats_greater_count = len(filter(lambda(stat, fischer): fischer[1] > stat[1], zip(chi_2_stats, fischer_stats)))
fischer_stats_less_count = len(filter(lambda(stat, fischer): fischer[1] < stat[1], zip(chi_2_stats, fischer_stats)))
print "A number of cells where p-value is less than 0.05 for Fischer criterion: %i" % len(filter(lambda stat: stat[1] < 0.05, fischer_stats))
stats_compare_plot(chi_2_corrected_stats, fischer_stats)
print "A number of Fischer stats where p-value is greater than p-value chi-square with correction is %i. Percentage: %.2f" % (fischer_stats_greater_count_c, float(fischer_stats_greater_count_c)/len(fischer_stats)*100)
print "A number of Fischer stats where p-value is less than p-value chi-square with correction is %i. Percentage: %.2f" % (fischer_stats_less_count_c, float(fischer_stats_less_count_c)/len(fischer_stats)*100)
stats_compare_plot(chi_2_stats, fischer_stats)
print "A number of Fischer stats where p-value is greater than p-value chi-square is %i. Percentage: %.2f" % (fischer_stats_greater_count, float(fischer_stats_greater_count)/len(fischer_stats)*100)
print "A number of Fischer stats where p-value is less than p-value chi-square is %i. Percentage: %.2f" % (fischer_stats_less_count, float(fischer_stats_less_count)/len(fischer_stats)*100)
avg_diff = np.average(map(lambda (chi2_stat, fischer_stat): fischer_stat[1]-chi2_stat[1], zip(chi_2_stats,fischer_stats)))
print "Average difference between chi2 stat and fischer stat is %.4f" % avg_diff
# Точный критерий Фишера на наших данных дает значения достигаемого уровня значимости в среднем меньшие, чем xи-квадрат с поправкой Йетса
# Точный критерий Фишера на наших данных дает значения достигаемого уровня значимости в среднем значительно большие, чем xи-квадрат без поправки
# Точный критерий Фишера всегда лучше, чем критерий xи-квадрат, потому что не использует аппроксимацию дискретного распределения непрерывным.
#   Однако при увеличении размера выборки его преимущества по сравнению с критерем xи-квадрат уменьшаются, в пределе достигая нуля.

# Давайте попробуем применить полученные знания о разных видах корреляции и ее применимости на практике.
# Рассмотрим пару признаков day_calls и mes_estim. Посчитайте корреляцию Пирсона между этими признаками на всех
# данных, ее значимость.
#%%
from scipy.stats import t
def student_criterion(correlation, count):
    corr = float(np.abs(correlation))
    cnt = float(count)
    stat = corr*np.sqrt(cnt-2.)/np.sqrt(1.-corr**2.)
    return (stat,(1.-t.cdf(stat, cnt-2))*2.)
#%%
pearson_corr = np.round(frame.corr(method="pearson")["day_calls"]["mes_estim"],4)
count = frame.shape[0]
pearson_stat,pearson_p_value = student_criterion(pearson_corr, count)

print "Pearson correlation for \"day_calls\" and \"mes_estim\" features: %.4f\tp-value: %.4f" % (np.round(pearson_corr,4), np.round(pearson_p_value,4))

# Отметьте все верные утверждения.
#    Корреляция Пирсона имеет отрицательный знак, и отличие корреляции от нуля на уровне доверия 0.05 значимо.


# Еще раз рассмотрим пару признаков day_calls и mes_estim. Посчитайте корреляцию Спирмена между этими признаками на
# всех данных, ее значимость.
#%%
spearman_corr = frame.corr(method="spearman")["day_calls"]["mes_estim"]
spearman_stat,spearman_p_value = student_criterion(spearman_corr, count)
print "Spearman correlation for \"day_calls\" and \"mes_estim\" features: %.4f\tp-value: %.4f" % (np.round(spearman_corr,4),np.round(spearman_p_value,4))
# Отметьте все верные утверждения.
#    Корреляция Спирмена имеет положительный знак, и отличие корреляции от нуля на уровне доверия 0.05 значимо.


# Как можно интерпретировать полученные значения коэффициентов корреляции и достигаемые уровни значимости при проверки
# гипотез о равенстве нулю этих коэффициентов?
#    Посчитанные корреляции и их значимости говорят лишь о том, что необходимо взглянуть на данные глазами и
#      попытаться понять, что приводит к таким (противоречивым?) результатам.


# Посчитайте значение коэффицента корреляции Крамера между двумя признаками: штатом (state) и оттоком пользователей
# (churn) для всех пользователей, которые находились в контрольной группе (treatment=1). Что можно сказать о
# достигаемом уровне значимости при проверке гипотезы о равенство нулю этого коэффициента?
#%%
from scipy.stats import chi2_contingency
contingency_table = control_states_pivot.as_matrix()
def v_Cramer_correlation(table):
    chi_stat = chi2_contingency(table)[0]
    k_min = np.min(table.shape)
    n = np.sum(table)
    return np.sqrt(chi_stat/(n*(k_min-1)))
cramer_correlation = v_Cramer_correlation(contingency_table)
cramer_p_value = chi2_contingency(contingency_table)[1]
print "V-Cramer correlation for \"state\" and \"churn\" features is %.4f\tp-value: %.4f" % (np.round(cramer_correlation,4), np.round(cramer_p_value, 4))
#    Для вычисления коэффициента Крамера используется значение статистики xи-квадрат, на которую мы не можем
#      положиться применительно к нашим данным.

# Вы прослушали большой курс и к текущему моменту обладете достаточными знаниями, чтобы попытаться самостоятельно
# выбрать нужный метод/инструмент/статистический критерий и сделать правильное заключение.
# В этой части задания вам нужно будет самостоятельно решить, с помощью каких методов можно провести анализ
# эффективности удержания (churn) с помощью раличных методов (treatment = 0, treatment = 2) относительно контрольной
# группы пользователей (treatment = 1).
# Что можно сказать об этих двух методах (treatment = 0, treatment = 2)? Одинаковы ли они с точки зрения эффективности?
# Каким бы методом вы бы посоветовали воспользоваться компании?
# Не забудьте про поправку на множественную проверку! И не пользуйтесь односторонними альтернативами, поскольку вы не
# знаете, к каким действительно последствиям приводят тестируемые методы (treatment = 0, treatment = 2)!
#%%
from statsmodels.stats.proportion import proportion_confint
strategy_0_group = frame[frame["treatment"] == 0]
strategy_0_group_left = strategy_0_group[strategy_0_group["churn"] == "True."]
strategy_2_group = frame[frame["treatment"] == 2]
strategy_2_group_left = strategy_2_group[strategy_2_group["churn"] == "True."]
control_group_left = control_group[control_group["churn"] == "True."]
str_0_left_count = float(len(strategy_0_group_left))
str_0_count = float(len(strategy_0_group))
strategy_0_proportion = str_0_left_count/str_0_count
str_2_left_count = float(len(strategy_2_group_left))
str_2_count = float(len(strategy_2_group))
strategy_2_proportion = str_2_left_count/str_2_count
control_left_count = float(len(control_group_left))
control_count = float(len(control_group))
control_proportion = control_left_count/control_count
str_0_confint = proportion_confint(str_0_left_count, str_0_count, alpha=0.05, method="normal")
str_2_confint = proportion_confint(str_2_left_count, str_2_count, alpha=0.05, method="normal")
control_confint = proportion_confint(control_left_count, control_count, alpha=0.05, method="normal")
print "Strategy 0 proportion: %.4f\tConfidential interval: %s" % (strategy_0_proportion,str_0_confint)
print "Strategy 2 proportion: %.4f\tConfidential interval: %s" % (strategy_2_proportion,str_2_confint)
print "Control proportion: %.4f\tConfidential interval: %s" % (control_proportion,control_confint)

#%%
def proportions_diff_z_stat_ind(sample1, sample2):
    n1 = len(sample1)
    n2 = len(sample2)
    
    p1 = float(sum(sample1)) / n1
    p2 = float(sum(sample2)) / n2 
    P = float(p1*n1 + p2*n2) / (n1 + n2)
    
    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))

def proportions_diff_z_test(sample1, sample2, alternative = "two-sided"):
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("alternative not recognized\n"
                         "should be \"two-sided\", \"less\" or \"greater\"")
    z_stat = proportions_diff_z_stat_ind(sample1, sample2)
    p_value = 0
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    elif alternative == 'less':
        p_value = stats.norm.cdf(z_stat)
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_stat)

    return (z_stat,p_value)
#%%
str_0_left_churns = map(lambda x: 1. if x == "True." else 0., strategy_0_group["churn"].as_matrix())
str_2_left_churns = map(lambda x: 1. if x == "True." else 0., strategy_2_group["churn"].as_matrix())
control_left_churns = map(lambda x: 1. if x == "True." else 0., control_group["churn"].as_matrix())
str_0_control_test = proportions_diff_z_test(
    str_0_left_churns,
    control_left_churns)
str_0_str_2_test = proportions_diff_z_test(
    str_0_left_churns,
    str_2_left_churns
)
str_2_control_test = proportions_diff_z_test(
    str_2_left_churns,
    control_left_churns
)
print str_0_control_test
print str_2_control_test
print str_0_str_2_test
#%%
from statsmodels.sandbox.stats.multicomp import multipletests
p_values = multipletests(
    [str_0_control_test[1], str_2_control_test[1], str_0_str_2_test[1]],
    alpha = 0.05,
    method = "fdr_bh")[1]
print "Strategy 0 vs control strategy p-value:%.4f" % p_values[0]
print "Strategy 2 vs control strategy p-value:%.4f" % p_values[1]
print "Strategy 0 vs strategy 2 p-value:%.4f" % p_values[2]
#    treatment = 2 статистически значимо отличается от контрольной группы treatment = 1
#    Отличие между treatment = 0 и treatment = 2 относительно влияния на уровень churn статистически незначимо.
