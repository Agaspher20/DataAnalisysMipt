# Давайте проанализируем данные опроса 4361 женщин из Ботсваны:
#%%
import pandas as pd
frame = pd.read_csv("..\..\Data\\botswana.tsv", sep="\t", header=0)
frame.head()
# О каждой из них мы знаем:
#    сколько детей она родила (признак ceb)
#    возраст (age)
#    длительность получения образования (educ)
#    религиозная принадлежность (religion)
#    идеальное, по её мнению, количество детей в семье (idlnchld)
#    была ли она когда-нибудь замужем (evermarr)
#    возраст первого замужества (agefm)
#    длительность получения образования мужем (heduc)
#    знает ли она о методах контрацепции (knowmeth)
#    использует ли она методы контрацепции (usemeth)
#    живёт ли она в городе (urban)
#    есть ли у неё электричество, радио, телевизор и велосипед (electric, radio, tv, bicycle)
# Давайте научимся оценивать количество детей ceb по остальным признакам.
# Сколько разных значений принимает признак religion?
#%%
print "Religion takes %i different values" % len(set(frame.religion.as_matrix()))

# Во многих признаках есть пропущенные значения. Сколько объектов из 4361 останется, если выбросить все, содержащие пропуски? 
#%%
print "Fully filled objects count: %i" % frame.dropna(axis=0, how="any").shape[0]

# В разных признаках пропуски возникают по разным причинам и должны обрабатываться по-разному.
# Например, в признаке agefm пропуски стоят только там, где evermarr=0, то есть, они соответствуют женщинам, никогда не выходившим замуж.
# Таким образом, для этого признака NaN соответствует значению "не применимо".
# В подобных случаях, когда признак x1 на части объектов в принципе не может принимать никакие значения, рекомендуется поступать так:
#   создать новый бинарный признак
#       x2=1, если x1='не применимо', иначе x2=0;
#   заменить "не применимо" в x1 на произвольную константу c, которая среди других значений x1 не встречается.
# Теперь, когда мы построим регрессию на оба признака и получим модель вида
#       y=β0+β1x1+β2x2,
# на тех объектах, где x1 было измерено, регрессионное уравнение примет вид
#       y=β0+β1x,
# а там, где x1 было "не применимо", получится
#       y=β0+β1c+β2.
# Выбор c влияет только на значение и интерпретацию β2, но не β1.
# Давайте используем этот метод для обработки пропусков в agefm и heduc.
#   Создайте признак nevermarr, равный единице там, где в agefm пропуски.
#   Удалите признак evermarr — в сумме с nevermarr он даёт константу, значит, в нашей матрице X будет мультиколлинеарность.
#   Замените NaN в признаке agefm на cagefm=0.
#   У объектов, где nevermarr = 1, замените NaN в признаке heduc на cheduc1=−1 (ноль использовать нельзя, так как он уже встречается у некоторых объектов выборки).
# Сколько осталось пропущенных значений в признаке heduc?
#%%
import numpy as np
frame["nevermarr"] = frame["agefm"].apply(lambda x: 1 if np.isnan(x) else 0)
frame.drop("evermarr", axis=1, inplace=True)
frame["agefm"] = frame["agefm"].apply(lambda x: 0 if np.isnan(x) else x)
frame.loc[frame["nevermarr"] == 1, "heduc"] = -1
#%%
print "Skipped heduc values count: %i" % (pd.isnull(frame["heduc"]).sum())

# Избавимся от оставшихся пропусков.
# Для признаков idlnchld, heduc и usemeth проведите операцию, аналогичную предыдущей:
#   создайте индикаторы пропусков по этим признакам (idlnchld_noans, heduc_noans, usemeth_noans),
#   замените пропуски на нехарактерные значения (cidlnchld=−1, cheduc2=−2 (значение -1 мы уже использовали), cusemeth=−1).
# Остались только пропуски в признаках knowmeth, electric, radio, tv и bicycle.
# Их очень мало, так что удалите объекты, на которых их значения пропущены.
# Какого размера теперь наша матрица данных? Умножьте количество строк на количество всех столбцов (включая отклик ceb).
#%%
frame["idlnchld_noans"] = frame["idlnchld"].apply(lambda x: 1 if np.isnan(x) else 0)
frame["heduc_noans"] = frame["heduc"].apply(lambda x: 1 if np.isnan(x) else 0)
frame["usemeth_noans"] = frame["usemeth"].apply(lambda x: 1 if np.isnan(x) else 0)
frame.loc[frame["idlnchld_noans"] == 1, "idlnchld"] = -1
frame.loc[frame["heduc_noans"] == 1, "heduc"] = -2
frame.loc[frame["usemeth_noans"] == 1, "usemeth"] = -1
frame = frame.dropna(axis=0, how="any")
print "Fully filled objects count after child count, use method and husband education preprocessing: %i" % frame.shape[0]
print "Matrix size: %i" % (frame.shape[0] * frame.shape[1])

# Постройте регрессию количества детей ceb на все имеющиеся признаки методом smf.ols, как в разобранном до этого примере.
# Какой получился коэффициент детерминации R2? Округлите до трёх знаков после десятичной точки.
#%%
frame.head()
#%%
import statsmodels.formula.api as smf
m1 = smf.ols("ceb ~ age + educ + religion + idlnchld + knowmeth + usemeth +"\
                "agefm + heduc + urban + electric + radio + tv + bicycle +"\
                "nevermarr + idlnchld_noans + heduc_noans + usemeth_noans",
             data=frame)
fitted = m1.fit()
print fitted.summary()

# Обратите внимание, что для признака religion в модели автоматически создалось несколько бинарных фиктивных переменных. Сколько их?
# Проверьте критерием Бройша-Пагана гомоскедастичность ошибки в построенной модели. Выполняется ли она?
#%%
import statsmodels.stats.api as sms
print "Breusch-Pagan test: p=%f" % sms.het_breuschpagan(fitted.resid, fitted.model.exog)[1]

# Удалите из модели незначимые признаки religion, radio и tv. Проверьте гомоскедастичность ошибки, при необходимости сделайте поправку Уайта.
# Не произошло ли значимого ухудшения модели после удаления этой группы признаков? Проверьте с помощью критерия Фишера.
# Чему равен его достигаемый уровень значимости? Округлите до четырёх цифр после десятичной точки.
# Если достигаемый уровень значимости получился маленький, верните все удалённые признаки; если он достаточно велик, оставьте модель без религии, тв и радио.
#%%
m2 = smf.ols("ceb ~ age + educ + idlnchld + knowmeth + usemeth +"\
                "agefm + heduc + urban + electric + bicycle +"\
                "nevermarr + idlnchld_noans + heduc_noans + usemeth_noans",
             data=frame)
fitted2 = m2.fit(cov_type="HC1")
comparison_result_1_2 = fitted.compare_f_test(fitted2)
print "F=%.4f, p=%.4f, k1=%.4f" % (np.round(comparison_result_1_2[0],4), np.round(comparison_result_1_2[1],4), np.round(comparison_result_1_2[2],4))

#%%
print fitted2.summary()
# Признак usemeth_noans значим по критерию Стьюдента, то есть, при его удалении модель значимо ухудшится.
# Но вообще-то отдельно его удалять нельзя: из-за того, что мы перекодировали пропуски в usemeth произвольно выбранным значением cusemeth=−1,
# удалять usemeth_noans и usemeth можно только вместе.
# Удалите из текущей модели usemeth_noans и usemeth. Проверьте критерием Фишера гипотезу о том, что качество модели не ухудшилось.
# Введите номер первой значащей цифры в достигаемом уровне значимости (например, если вы получили 5.5×10−8, нужно ввести 8).
# Если достигаемый уровень значимости получился маленький, верните удалённые признаки; если он достаточно велик, оставьте модель без usemeth и usemeth_noans.
#%%
m3 = smf.ols("ceb ~ age + educ + idlnchld + knowmeth + "\
                "agefm + heduc + urban + electric + bicycle +"\
                "nevermarr + idlnchld_noans + heduc_noans",
             data=frame)
fitted3 = m3.fit(cov_type="HC1")
comparison_result = fitted2.compare_f_test(fitted3)
print "F=%f, p=%f, k1=%f" % comparison_result
print comparison_result[1]

# Посмотрите на доверительные интервалы для коэффициентов итоговой модели
# (не забудьте использовать поправку Уайта, если есть гетероскедастичность ошибки) и выберите правильные выводы.
#%%
print fitted2.summary()
