# Задача заключается в том, чтобы применить предобученную на imagenet нейронную сеть на практической задаче классификации
# автомобилей.
# Учиться применять нейронные сети для анализа изображений мы будем на библиотеке TensorFlow.
# Это известный опенсорсный проект, разработанный инженерами Google Brain Team.
# Чтобы освоить компьютерное зрение (или другие интересные задачи из области ML и AI), так или иначе придётся научиться
# работать с библиотеками нейронных сетей, линуксом и виртуальными серверами. Например, для более масштабных практических
# задач, крайне необходимы сервера с GPU, а с ними уже локально работать не получится.
# Помимо tensorflow, потребуется библиотека scipy.
# Если вы уже работали с Anaconda и/или выполняли задания в нашей специализации, то она должна присутствовать. 

# Данные это часть выборки Cars Dataset
# Исходный датасет содержит 16,185 изображений автомобилей, принадлежащих к 196 классам.
# Данные разделены на 8,144 тренировочных и 8,041 тестовых изображений, при этом каждый класс разделён приблизительно
# поровну между тестом и трейном.
# Все классы уровня параметров Марка, Год, Модель и др. (например, 2012 Tesla Model S or 2012 BMW M3 coupe).
# В нашем же случае в train 204 изображения, и в test — 202 изображения.
#%%
# Импортируем всё, что нам нужно для работы.
import sys
sys.path.append("DataAnalisysMipt\\Python\\5. DataAnalysisApplications")
from task2_1_imagenet_classes import class_names
from task2_1_model import vgg16
import task2_1_result_functions as rf
import glob
import os
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import sys
from sklearn.svm import SVC
#%%
# Инициируем TF сессию, и инициализируем модель. На этом шаге модель загружает веса. Веса - это 500Мб в сжатом виде
# и ~2.5Гб в памяти, процесс их загрузки послойно выводится ниже этой ячейки, и если вы увидите этот вывод ещё раз - 
# у вас неистово кончается память. Остановитесь. Также, не запускайте эту ячейку на выполнение больше одного раза
# за запуск ядра Jupyter.
sess = tf.Session()
imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = vgg16(imgs, "DataAnalisysMipt\\Data\\automobiles\\vgg16_weights.npz", sess)
# Все ячейки выше не нуждаются в модификации для выполнения задания, и необходимы к исполнению только один раз,
# в порядке следования. Повторный запуск ячейки с инициализацией модели будет сжирать память. Вы предупреждены.
#%%
# Задание 1.
# Для начала нужно запустить готовую модель vgg16, предобученную на imagenet.
# Модель обучена с помощью caffe и сконвертирована в формат tensorflow - vgg16_weights.npz.
# Скрипт, иллюстрирующий применение этой модели к изображению, возвращает топ-5 классов из imagenet и уверенность в этих
# классах.
# Задание: Загрузите уверенность для первого класса для изображения train/00002.jpg с точностью до 1 знака после запятой
# в файл с ответом.
probabilities, predictions = rf.process_image(
    "DataAnalisysMipt\\Data\\automobiles\\train\\00002.jpg",
    sess,
    vgg)
for prediction in predictions[0:5]:
    print(class_names[prediction], probabilities[prediction])
#%%
print(
    "First class is %s. Its probability is %.1f"
    % (class_names[predictions[0]], np.round(probabilities[predictions[0]], 1)))
rf.save_answer_num("DataAnalisysMipt\\Results\\pa_5_2_1_1.txt",
                   np.round(probabilities[predictions[0]], 1))

# Задание 2.
# Научитесь извлекать fc2 слой.
# Для этого нужно модифицировать process_image, чтобы вместо последнего слоя извлекались выходы fc2.
# Задание:
# Посчитайте fc2 для картинки train/00002.jpg.
# Запишите первые 20 компонент (каждое число с новой строки, т.е. в загружаемом файле должно получиться 20 строк).
#%%
def process_image_fc2(fname, sess, vgg):
    """Функция обработки отдельного изображения"""
    # Печатает метки TOP-5 классов и уверенность модели в каждом из них.
    img1 = imread(fname, mode="RGB")
    img1 = imresize(img1, (224, 224))

    prob = sess.run(vgg.fc2, feed_dict={vgg.imgs: [img1]})[0]
    return (prob, np.argsort(prob)[::-1])
#%%
probabilities_fc2, predictions_fc2 = process_image_fc2(
    "DataAnalisysMipt\\Data\\automobiles\\train\\00002.jpg",
    sess,
    vgg)
save_answer_array("DataAnalisysMipt\\Results\\pa_5_2_1_2.txt", probabilities_fc2[0:20])

# Задание 3.
# Теперь необходимо дообучить классификатор на нашей базе.
# В качестве бейзлайна предлагается воспользоваться классификатором svm из пакета scipy.
#   Модифицировать функцию get_features и добавить возможность вычислять fc2. (Аналогично второму заданию).
#   Применить get_feautures, чтобы получить X_test и Y_test.
#   Воспользоваться классификатором SVC с random_state=0.
#%%
def get_features(folder, ydict):
    """Функция, возвращающая признаковое описание для каждого файла jpg в заданной папке"""
    paths = glob.glob(folder)
    X = np.zeros((len(paths), 4096))
    Y = np.zeros(len(paths))

    for i, img_name in enumerate(paths):
        print(img_name)
        base = os.path.basename(img_name)
        Y[i] = ydict[base]

        img1 = imread(img_name, mode="RGB")
        img1 = imresize(img1, (224, 224))
        # Здесь ваш код. Нужно получить слой fc2
        fc2 = sess.run(vgg.fc2, feed_dict={vgg.imgs: [img1]})[0]
        X[i, :] = fc2
    return X, Y
#%%
def process_folder(folder):
    """ Функция обработки папки. """
    # Ожидается, что в этой папке лежит файл results.txt с метками классов,
    # и имеются подразделы train и test с jpg файлами.
    ydict = rf.load_txt(os.path.join(folder, "results.txt"))
    X, Y = get_features(os.path.join(folder, "train/*jpg"), ydict)
    # Ваш код здесь.
    X_test, Y_test = get_features(os.path.join(folder, "test/*jpg"), ydict)
    # Ваш код здесь.
    clf = SVC(random_state=0).fit(X, Y)
    Y_test_pred = clf.predict(X_test)
    return (Y_test, Y_test_pred)
#%%
Y_test, Y_test_pred = process_folder("DataAnalisysMipt\\Data\\automobiles")
print(sum(Y_test == Y_test_pred)) # Число правильно предсказанных классов
rf.save_answer_num("DataAnalisysMipt\\Results\\pa_5_2_1_3.txt",
                   sum(Y_test == Y_test_pred))
