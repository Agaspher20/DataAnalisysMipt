"""Функции сохранения ответов"""

from scipy.misc import imread, imresize
import numpy as np
from task2_1_imagenet_classes import class_names

def save_answer_num(fname, number):
    """Функция сохранения в файл ответа, состоящего из одного числа"""
    with open(fname, "w") as fout:
        fout.write(str(number))

def save_answer_array(fname, array):
    """Функция сохранения в файл ответа, представленного массивом"""
    with open(fname, "w") as fout:
        fout.write("\n".join([str(el) for el in array]))

def load_txt(fname):
    """Загрузка словаря из текстового файла."""
    # Словарь у нас используется для сохранения меток классов в выборке data.
    line_dict = {}
    for line in open(fname):
        fname, class_id = line.strip().split()
        line_dict[fname] = class_id

    return line_dict

def process_image(fname, sess, vgg):
    """Функция обработки отдельного изображения"""
    # Печатает метки TOP-5 классов и уверенность модели в каждом из них.
    img1 = imread(fname, mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    return (prob, np.argsort(prob)[::-1])
