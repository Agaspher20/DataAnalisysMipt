## Градиентный бустинг своими руками
# В этом задании будет использоваться датасет boston из sklearn.datasets. Оставьте последние 25% объектов для
# контроля качества, разделив X и y на X_train, y_train и X_test, y_test.
# Целью задания будет реализовать простой вариант градиентного бустинга над регрессионными деревьями для случая
# квадратичной функции потерь.
#%%
import numpy as np
from sklearn import datasets
dataset = datasets.load_boston()
dataset_size = len(dataset.data)
train_size = int(0.75*dataset_size)
X = dataset.data
y = dataset.target
X_train = X[0:train_size]
y_train = y[0:train_size]
X_test = X[train_size:dataset_size]
y_test = y[train_size:dataset_size]

def write_answer(fileName, answer):
    with open("..\..\Results\gradientBoosting_" + fileName + ".txt", "w") as fout:
        fout.write(str(answer))

## Задание 1
# Как вы уже знаете из лекций, бустинг - это метод построения композиций базовых алгоритмов с помощью
# последовательного добавления к текущей композиции нового алгоритма с некоторым коэффициентом. Градиентный
# бустинг обучает каждый новый алгоритм так, чтобы он приближал антиградиент ошибки по ответам композиции на
# обучающей выборке. Аналогично минимизации функций методом градиентного спуска, в градиентном бустинге мы
# подправляем композицию, изменяя алгоритм в направлении антиградиента ошибки. Воспользуйтесь формулой из
# лекций, задающей ответы на обучающей выборке, на которые нужно обучать новый алгоритм (фактически это лишь
# чуть более подробно расписанный градиент от ошибки), и получите частный ее случай, если функция потерь
# L - квадрат отклонения ответа композиции a(x) от правильного ответа y на данном x.

# Если вы давно не считали производную самостоятельно, вам поможет таблица производных элементарных функций
# (которую несложно найти в интернете) и правило дифференцирования сложной функции. После дифференцирования
# квадрата у вас возникнет множитель 2 — т.к. нам все равно предстоит выбирать коэффициент, с которым будет
# добавлен новый базовый алгоритм, проигноируйте этот множитель при дальнейшем построении алгоритма.
#%%
def Ls(p, y): return y-p

## Задание 2
# Заведите массив для объектов DecisionTreeRegressor (будем их использовать в качестве базовых алгоритмов) и
# для вещественных чисел (это будут коэффициенты перед базовыми алгоритмами). В цикле обучите последовательно
# 50 решающих деревьев с параметрами max_depth=5 и random_state=42 (остальные параметры - по умолчанию). В
# бустинге зачастую используются сотни и тысячи деревьев, но мы ограничимся 50, чтобы алгоритм работал быстрее,
# и его было проще отлаживать (т.к. цель задания разобраться, как работает метод). Каждое дерево должно
# обучаться на одном и том же множестве объектов, но ответы, которые учится прогнозировать дерево, будут
# меняться в соответствие с полученным в задании 1 правилом. Попробуйте для начала всегда брать коэффициент
# равным 0.9. Обычно оправдано выбирать коэффициент значительно меньшим - порядка 0.05 или 0.1, но т.к. в нашем
# учебном примере на стандартном датасете будет всего 50 деревьев, возьмем для начала шаг побольше.

# В процессе реализации обучения вам потребуется функция, которая будет вычислять прогноз построенной на данный
# момент композиции деревьев на выборке X:
#%%
from sklearn.tree import DecisionTreeRegressor
def buildGradientBoostingCoefConst(X, y, tree_depth=5, rand_state=42,
                                   trees_count=50, const_coef=0.9):
    base_algorithms_list = [
        DecisionTreeRegressor(max_depth=tree_depth, random_state=rand_state).fit(X, y)]
    coefficients_list = [const_coef]
    def gbm_predict(X):
        return [sum([coeff * algo.predict([x])[0]
                     for algo, coeff in zip(base_algorithms_list, coefficients_list)])
                for x in X]
    for i in xrange(trees_count-1):
        predictions = gbm_predict(X)
        reg = DecisionTreeRegressor(max_depth=tree_depth, random_state=rand_state)
        reg = reg.fit(X, Ls(predictions, y))
        base_algorithms_list.append(reg)
        coefficients_list.append(const_coef)
    return gbm_predict
# (считаем, что
#   base_algorithms_list - список с базовыми алгоритмами,
#   coefficients_list - список с коэффициентами перед алгоритмами)

# Эта же функция поможет вам получить прогноз на контрольной выборке и оценить качество работы вашего алгоритма
# с помощью mean_squared_error в sklearn.metrics. 
# Возведите результат в степень 0.5, чтобы получить RMSE. Полученное значение RMSE — ответ в пункте 2.
#%%
from sklearn.metrics import mean_squared_error as mse
predictor = buildGradientBoostingCoefConst(X_train, y_train)
final_predictions = predictor(X_test)
answer_2 = np.sqrt(mse(y_test, final_predictions))
write_answer("2", answer_2)
answer_2

## Задание 3
# Вас может также беспокоить, что двигаясь с постоянным шагом, вблизи минимума ошибки ответы на обучающей
# выборке меняются слишком резко, перескакивая через минимум. 
# Попробуйте уменьшать вес перед каждым алгоритмом с каждой следующей итерацией по формуле 0.9 / (1.0 + i),
# где i - номер итерации (от 0 до 49). Используйте качество работы алгоритма как ответ в пункте 3.
# В реальности часто применяется следующая стратегия выбора шага: как только выбран алгоритм, подберем
# коэффициент перед ним численным методом оптимизации таким образом, чтобы отклонение от правильных ответов
# было минимальным. Мы не будем предлагать вам реализовать это для выполнения задания, но рекомендуем
# попробовать разобраться с такой стратегией и реализовать ее при случае для себя.
#%%
def buildGradientBoosting(X, y, tree_depth=5, rand_state=42,
                                   trees_count=50, initial_coef=0.9):
    base_algorithms_list = [
        DecisionTreeRegressor(max_depth=tree_depth, random_state=rand_state).fit(X, y)]
    coefficients_list = [initial_coef]
    def gbm_predict(X):
        return [sum([coeff * algo.predict([x])[0]
                     for algo, coeff in zip(base_algorithms_list, coefficients_list)])
                for x in X]
    for i in xrange(1,trees_count):
        predictions = gbm_predict(X)
        reg = DecisionTreeRegressor(max_depth=tree_depth, random_state=rand_state)
        reg = reg.fit(X, Ls(predictions, y))
        base_algorithms_list.append(reg)
        coefficients_list.append(initial_coef/(1.0+i))
    return gbm_predict
#%%
predictor_3 = buildGradientBoosting(X_train, y_train)
final_predictions_3 = predictor_3(X_test)
answer_3 = np.sqrt(mse(y_test, final_predictions_3))
write_answer("3", answer_3)
answer_3
## Задание 4
# Реализованный вами метод - градиентный бустинг над деревьями - очень популярен в машинном обучении. Он
# представлен как в самой библиотеке sklearn, так и в сторонней библиотеке XGBoost, которая имеет свой
# питоновский интерфейс. На практике XGBoost работает заметно лучше GradientBoostingRegressor из sklearn, но
# для этого задания вы можете использовать любую реализацию.
# Исследуйте, переобучается ли градиентный бустинг с ростом числа итераций (и подумайте, почему), а также с
# ростом глубины деревьев. На основе наблюдений выпишите через пробел номера правильных из приведенных ниже
# утверждений в порядке возрастания номера (это будет ответ в п.4):
# 1. С увеличением числа деревьев, начиная с некоторого момента, качество работы градиентного бустинга не
#    меняется существенно.
# 2. С увеличением числа деревьев, начиная с некоторого момента, градиентный бустинг начинает переобучаться.
#%%
def calcMse(treesCount=50, treesDepth=5):
    predictor = buildGradientBoosting(X_train, y_train, trees_count=treesCount, tree_depth=treesDepth)
    predictions = predictor(X_test)
    return np.sqrt(mse(y_test, predictions))
treesCounts = range(50, 100, 5)
mse_treesCount = [calcMse(treesCount=trees_count) for trees_count in treesCounts]
#%%
from matplotlib import pyplot as plt
plt.plot(treesCounts, mse_treesCount)
# 3. С ростом глубины деревьев, начиная с некоторого момента, качество работы градиентного бустинга на тестовой
#    выборке начинает ухудшаться.
# 4. С ростом глубины деревьев, начиная с некоторого момента, качество работы градиентного бустинга перестает
#    существенно изменяться
#%%
depths = range(5, 10)
mse_depth = [calcMse(treesDepth = depth) for depth in depths]
#%%
plt.plot(depths, mse_depth)
#%%
write_answer("4", " ".join(map(str, [2, 3])))

## Задание 5
# Сравните получаемое с помощью градиентного бустинга качество с качеством работы линейной регрессии. 
# Для этого обучите LinearRegression из sklearn.linear_model (с параметрами по умолчанию) на обучающей выборке
# и оцените для прогнозов полученного алгоритма на тестовой выборке RMSE. Полученное качество - ответ в пункте 5.
# В данном примере качество работы простой модели должно было оказаться хуже, но не стоит забывать, что так
# бывает не всегда. В заданиях к этому курсу вы еще встретите пример обратной ситуации.
#%%
from sklearn.linear_model import LinearRegression
linear = LinearRegression().fit(X_train, y_train)
linear_predictions = linear.predict(X_test)
linear_mse = mse(y_test, linear_predictions)
answer_5 = np.sqrt(linear_mse)
write_answer("5", answer_5)
answer_5
