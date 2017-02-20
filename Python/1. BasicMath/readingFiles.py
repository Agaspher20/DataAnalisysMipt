filePath = '..\Data\example_reading_files.txt'
file_object = open(filePath, 'r')
print type(file_object)
print file_object.read()
file_object.close()

file_object = open(filePath, 'r')
print file_object.readline()
print file_object.readline()

for line in file_object:
    print line.strip()
file_object.close()

file_object = open(filePath, 'r')
data_list = list(file_object)
file_object.close()

for line in data_list:
    print line

file_object = open(filePath, 'r')
data_list = file_object.readlines()
for line in data_list:
    print line.strip()
file_object.close()

file_object = open('..\Data\example_koi_8.txt', 'r')
for line in file_object: print line.strip()
file_object.close()

import codecs
file_object = codecs.open('D:\\example_koi_8.txt', 'r', encoding='koi8-r')
print file_object.read()
