# Функция сохранения в файл ответа, состоящего из одного числа
def save_answerNum(fname,number):
    with open(fname,"w") as fout:
        fout.write(str(number))
# Функция сохранения в файл ответа, представленного массивом
def save_answerArray(fname,array):
    with open(fname,"w") as fout:
        fout.write(" ".join([str(el) for el in array]))
# Загрузка словаря из текстового файла. Словарь у нас используется для сохранения меток классов в выборке data.
def load_txt(fname):
    line_dict = {}
    for line in open(fname):
        fname, class_id = line.strip().split()
        line_dict[fname] = class_id

    return line_dict
# Функция обработки отдельного изображения, печатает метки TOP-5 классов и уверенность модели в каждом из них.
def process_image(fname):
    img1 = imread(fname, mode='RGB')
    img1 = imresize(img1, (224, 224))
    
    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])