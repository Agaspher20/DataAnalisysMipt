x = 5
print x
print type(x)
a = 4 + 5
b = 4 * 5
c = 5 / 4
print a, b, c
print -5/4
print -(5/4)
x = 5 * 1000000 * 1000000 * 1000000 * 1000000 + 1
print type(x)
y = 5.7
a = 4.2 + 5.1
b = 4.2 * 5.1
c = 5.0 / 4.0
print a, b, c
a = 5
b = 4
print float(a)/float(b)

print 5./4
print 5/4.

a = True
b = False
print a
print type(a)

print b
print type(b)

print a + a
print a + b
print b + b

print int(a), int(b)

print True and False
print True and True
print False and False

print True or False
print True or True
print False or False

z = None
print z
print type(z)
print int(z)

x = 'abc'
print x
print type(x)

a = 'Ivan'
b = 'Ivanov'
s = a + ' ' + b

print s

print a.upper()
print a.lower()

print len(a)
print bool(a)
print bool('')
print int(a)

print a[0]
print a[1]
print a[0:3]
print a[0:4:2]

x = u'Abc'
print x
print type(x)

x = u'Эльвира Михайловна'
print x, type(x)
y = x.encode('utf-8')
print y, type(y)
z = y.decode('utf-8')
print z, type(z)
q = y.decode('cp1251')
print q, type(q)

print str(x)

print y[1:]
print len(y), type(y)
print len(x), type(x)

y = u'Иван Иванович'.encode('utf-8')
print y.decode('utf-8')
print y.decode('cp1251')

splitted_line = 'Ivanov Ivan Ivanovich'.split(' ')
print splitted_line
print type(splitted_line)
splitted_line = 'Иванов Иван Иванович'.split(' ')
print splitted_line
splitted_line = u'Иванов Иван Иванович'.split(' ')
print splitted_line

sealed_goods_count = [33450, 33195, 84123]
print sealed_goods_count
print type(sealed_goods_count)

income = ['Высокий', 'Средний', 'Высокий']
names = ['Элеонора Михайловна', 'Абрам Никифорович', 'Михаил Васильевич']
print income
print names
print ' '.join(income)

replaced = '_'.join('Hello world'.split(' '))
print replaced

features = ['Ivan ivanovich', 5, 13, True]

print features[0], features[1], features[3]
print features[0:5]
print features[:5]
print features[1:]
print features[2:5]
print features[:-1]
features.append('appended value')
print features
del features[-2]

features_tuple = ('Ivan ivanovich', 5, 13, True)
print features_tuple
print type(features_tuple)
features_tuple[2:5]
features_tuple.append('appended value')

names = {'Ivan', 'Petr', 'Konstantin'}
print type(names)
print 'Ivan' in names
print 'Michael' in names
names.add('Michael')
names.remove('Ivan')
names.add(['Vladimir', 'Vladimirovich'])

names.add(('Vladimir', 'Vladimirovich'))
print names

a = range(10000)
b = range(10000)
b = set(b)

print a[:5]
print a[-5:]

print 9999 in a
print 9999 in b

word_frequencies = dict()
word_frequencies['I'] = 1
word_frequencies['am'] = 1
word_frequencies['I'] += 1

print word_frequencies
print word_frequencies['I']

freqs = {'I': 2, 'am': 1}
print freqs
print type(word_frequencies)
print type(freqs)