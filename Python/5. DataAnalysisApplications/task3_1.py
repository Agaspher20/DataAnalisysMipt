""" Классификация текстов: спам-фильтр для SMS """

# В этом задании вам предстоит взять открытый датасет с SMS-сообщениями,
# размеченными на спам ("spam") и не спам ("ham"), построить на нем классификатор текстов на эти
# два класса, оценить его качество с помощью кросс-валидации, протестировать его работу на
# отдельных примерах, и посмотреть, что будет происходить с качеством, если менять параметры вашей
# модели.

# Задание
# Загрузите датасет.
# Выясните, что используется в качестве разделителей и как проставляются метки классов.
#%%
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score
sys.path.append("DataAnalisysMipt\\Python\\5. DataAnalysisApplications")
data = pd.read_csv(
    "DataAnalisysMipt\\Data\\SMSSpamCollection.txt",
    "\t",
    header=0,
    names=["label", "message"])
data.head()

# Подготовьте для дальнейшей работы два списка: список текстов в порядке их следования в датасете
# и список соответствующих им меток классов.
# В качестве метки класса используйте 1 для спама и 0 для "не спама".
#%%
labels, texts = zip(*[(row[0],row[1]) for row in data.as_matrix()])

# Используя sklearn.feature_extraction.text.CountVectorizer со стандартными настройками,
# получите из списка текстов матрицу признаков X.
#%%
vectorizer = CountVectorizer().fit(texts)
y = [0 if label == "ham" else 1 for label in labels]
X = vectorizer.transform(texts).toarray()

# Оцените качество классификации текстов с помощью LogisticRegression() с параметрами по умолчанию,
# используя sklearn.cross_validation.cross_val_score и посчитав среднее арифметическое качества на
# отдельных fold'ах. Установите random_state=2. Параметр cv задайте равным 10.
# В качестве метрики качества используйте f1-меру.
#%%
logistic_classifier = LogisticRegression(random_state=2)
logistic_classifier_score = cross_val_score(logistic_classifier, X, y, cv=10, scoring="f1")
#%%
def save_answer_num(fname, number):
    """Функция сохранения в файл ответа, состоящего из одного числа"""
    with open(fname, "w") as fout:
        fout.write(str(number))
#%%
# Получившееся качество - один из ответов, которые потребуются при сдаче задания.
# Ответ округлить до 1 знака после запятой.
logistic_classifier_mean_score = np.round(logistic_classifier_score.mean(), 1)
save_answer_num("DataAnalisysMipt\\Results\\pa_5_3_1_1.txt", logistic_classifier_mean_score)
print(
    "Mean cross val score for logistic regression classifier is %.1f"
    % logistic_classifier_mean_score)

# А теперь обучите классификатор на всей выборке и спрогнозируйте с его помощью класс для следующих
# сообщений:
#%%
unknown_messages = [
    "FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! Subscribe6GB",
    "FreeMsg: Txt: claim your reward of 3 hours talk time",
    "Have you visited the last lecture on physics?",
    "Have you visited the last lecture on physics? Just buy this book and you will have all materials! Only 99$",
    "Only 99$"]
#%%
log_classifier = LogisticRegression(random_state=2).fit(X, y)
unknown_features = vectorizer.transform(unknown_messages).toarray()
log_classifier_results = [log_classifier.predict(message) for message in unknown_features]
# Прогнозы классификатора (0 - не спам, 1 - спам), записанные через пробел, будут ответом в одном
# из вопросов ниже.
#%%
log_classifier_result_string = " ".join([str(res[0]) for res in log_classifier_results])
print("Default logistic regression results: %s" % (log_classifier_result_string))
save_answer_num("DataAnalisysMipt\\Results\\pa_5_3_1_2.txt", log_classifier_result_string)

# Задайте в CountVectorizer параметр ngram_range=(2,2), затем ngram_range=(3,3),
# затем ngram_range=(1,3).
#%%
ngram_ranges = [(2,2), (3,3), (1,3)]
def calculate_ngram_result(ngram_range, texts, vectorizer, model):
    vectorizer_ngram = vectorizer(ngram_range=ngram_range).fit(texts)
    X_ngram = vectorizer_ngram.transform(texts)
    # Во всех трех случаях измерьте получившееся в кросс-валидации значение f1-меры,
    return cross_val_score(model, X_ngram, y, cv=10, scoring="f1").mean()
ngram_results = [
    calculate_ngram_result(
        ngram_range,
        texts,
        CountVectorizer,
        LogisticRegression(random_state=2))
    for ngram_range in ngram_ranges]
# округлите до второго знака после точки, и выпишете результаты через пробел в том же порядке.
ngram_result_string = " ".join([str(np.round(res, 2)) for res in ngram_results])
print("Ngram results are: %s" % (ngram_result_string))
save_answer_num("DataAnalisysMipt\\Results\\pa_5_3_1_3.txt", ngram_result_string)
# В данном эксперименте мы пробовали добавлять в признаки n-граммы для разных диапазонов n -
# только биграммы, только триграммы, и, наконец, все вместе - униграммы, биграммы и триграммы.
# Обратите внимание, что статистики по биграммам и триграммам намного меньше,
# поэтому классификатор только на них работает хуже.
# В то же время это не ухудшает результат сколько-нибудь существенно,
# если добавлять их вместе с униграммами, т.к. за счет регуляризации линейный классификатор не
# склонен сильно переобучаться на этих признаках.

# Повторите аналогичный п.7 эксперимент, используя вместо логистической регрессии MultinomialNB().
# Обратите внимание, насколько сильнее (по сравнению с линейным классификатором) наивный Байес
# страдает от нехватки статистики по биграммам и триграммам.
# По какой-то причине обучение наивного байесовского классификатора через Pipeline происходит с
# ошибкой. Чтобы получить правильный ответ, отдельно посчитайте частоты слов и обучите
# классификатор.
#%%
ngram_bayes_results = [
    calculate_ngram_result(
        ngram_range,
        texts,
        CountVectorizer,
        MultinomialNB())
    for ngram_range in ngram_ranges]
# округлите до второго знака после точки, и выпишете результаты через пробел в том же порядке.
ngram_result_bayes_string = " ".join([str(np.round(res, 2)) for res in ngram_bayes_results])
print("Ngram Bayes results are: %s" % (ngram_result_bayes_string))
save_answer_num("DataAnalisysMipt\\Results\\pa_5_3_1_4.txt", ngram_result_bayes_string)

# Попробуйте использовать в логистической регрессии в качестве признаков Tf*idf из TfidfVectorizer
# на униграммах.
#%%
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1)).fit(texts)
X_tfidf = tfidf_vectorizer.transform(texts).toarray()
logistic_classifier_tfidf = LogisticRegression(random_state=2)
logistic_classifier_score_tfidf = cross_val_score(
    logistic_classifier_tfidf,
    X_tfidf,
    y,
    cv=10,
    scoring="f1")
#%%
logistic_classifier_mean_score_tfidf = logistic_classifier_score_tfidf.mean()
print("Count vectorizer score: %f" % (logistic_classifier_mean_score))
print("Tfidf vectorizer score: %f" % (logistic_classifier_mean_score_tfidf))

# Повысилось или понизилось качество на кросс-валидации по сравнению с CountVectorizer на
# униграммах?
# (напишите в файле с ответом 1, если повысилось, -1, если понизилось, и 0, если изменилось
# не более чем на 0.01).
#%%
score_diff = logistic_classifier_mean_score_tfidf - logistic_classifier_mean_score
if np.abs(score_diff) < 0.01:
    answer_9 = 0
else:
    answer_9 = 1 if score_diff > 0 else -1
save_answer_num("DataAnalisysMipt\\Results\\pa_5_3_1_5.txt", answer_9)
# Обратите внимание, что результат перехода к tf*idf не всегда будет таким - если вы наблюдаете
# какое-то явление на одном датасете, не надо сразу же его обобщать на любые данные.
