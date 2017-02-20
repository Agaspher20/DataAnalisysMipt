resultPath = '..\Results\FileToWriteIn.txt'
file_obj_write = open(resultPath, 'w')
input = 'String to write\n'
file_obj_write.write(input)
file_obj_write.close()

file_obj_read = open(resultPath, 'r')
print file_obj_read.read()
file_obj_read.close()

file_obj_write = open(resultPath, 'a')
input = 'String to write 2\n'
file_obj_write.write(input)
file_obj_write.close()

file_obj_read = open(resultPath, 'r')
print file_obj_read.read()
file_obj_read.close()

digits = map(lambda dig: str(dig) + '\n', range(1, 11))
print digits
file_obj_write = open(resultPath, 'w')
file_obj_write.writelines(digits)
file_obj_write.close()

file_obj_read = open(resultPath, 'r')
print file_obj_read.read()
file_obj_read.close()
