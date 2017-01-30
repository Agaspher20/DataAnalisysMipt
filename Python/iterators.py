x = True

if x:
    print 'Ok'
else:
    print 'Not ok'

for i in range(10):
    print i,

print range(2,5)

for i in [2,3,4]:
    print i,

type(range(2,5))

for i in xrange(5,14):
    print i,

type(xrange(2,5))

print xrange(2,5)

print [x**2 for x in range(1,11) if x%2 == 0]
w = [x**2 for x in range(1,11) if x%2 == 0]
print type(w)
g = (x**2 for x in range(1,11) if x%2 == 0)
print type(g)

x = True
s = 0
while x:
    s+=1
    if s%2 == 0:
        print 'Continue'
        continue
    print s
    if s > 10:
        break

def myrange(a, b):
    res = []
    s = a
    while s != b:
        res.append(s)
        s += 1
    return res
print myrange(2,7)

print [x**2 for x in range(10)]

def sq(x):
    return x**2
print map(sq, range(10))
print map(lambda x: x**2, range(10))
