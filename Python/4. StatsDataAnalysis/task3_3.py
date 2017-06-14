# Классификатор C4.5 и три его модификации:
#   с оптимизацией гиперпараметра m,
#   гиперпараметра cf
#   и с одновременной оптимизацией обоих гиперпараметров.
# Эти четыре классификатора сравнивались на 14 наборах данных.
# На каждом датасете был посчитан AUC каждого классификатора. Данные записаны в файле:
#%%
import pandas as pd
frame = pd.read_csv("AUCs.txt", sep="\t", header=0)
frame.head()
# Используя критерий знаковых рангов, проведите попарное сравнение каждого классификатора с каждым.
# Выберите два классификатора, различие между которыми наиболее статистически значимо.
#%%
from scipy import stats
classificators_matrix = frame.drop("Unnamed: 0", axis=1)
columns_count = len(classificators_matrix.columns)
min_pvalue = 1.
min_first = ""
min_second = ""
valueable_differencies_count = 0
comparison_result = []
for (i,first) in enumerate(classificators_matrix.columns):
    if i < columns_count:
        for j in range(i+1, columns_count):
            second = classificators_matrix.columns[j]
            first_matrix = classificators_matrix[first].as_matrix()
            second_matrix = classificators_matrix[second].as_matrix()
            wilcox = stats.wilcoxon(first_matrix, second_matrix)
            comparison_result.append([first+"_"+second, wilcox.statistic, wilcox.pvalue])
            if wilcox.pvalue < 0.05:
                valueable_differencies_count += 1
            if min_pvalue > wilcox.pvalue:
                min_pvalue = wilcox.pvalue
                min_first = first
                min_second = second
comparison_frame = pd.DataFrame(comparison_result, columns=["Names", "Statistic", "p-value"])
comparison_frame
#%%
print "\nMost different classificators: \"%s\" and \"%s\" with p-value: %f" % (min_first, min_second, min_pvalue)
# Сколько статистически значимых на уровне 0.05 различий мы обнаружили?
#%%
print "Statistically valuable differencies count: %i" % valueable_differencies_count

# Сравнивая 4 классификатора между собой, мы проверили 6 гипотез.
# Давайте сделаем поправку на множественную проверку. Начнём с метода Холма.
# Сколько гипотез можно отвергнуть на уровне значимости 0.05 после поправки этим методом?
#%%
from statsmodels.sandbox.stats.multicomp import multipletests 
reject_holm, p_corrected_holm, a1_holm, a2_holm = multipletests(comparison_frame["p-value"],
                                                                alpha = 0.05,
                                                                method = 'holm')
print "Hypothesis to reject after holm correction count: %i" % len(filter(lambda whether_reject: whether_reject, reject_holm))

# Сколько гипотез можно отвергнуть на уровне значимости 0.05 после поправки методом
# Бенджамини-Хохберга? 
#%%
reject_fdr, p_corrected_fdr, a1_fdr, a2_fdr = multipletests(comparison_frame["p-value"],
                                                            alpha = 0.05,
                                                            method = 'fdr_bh')
print "Hypothesis to reject after fdr correction count: %i" % len(filter(lambda whether_reject: whether_reject, reject_fdr))
