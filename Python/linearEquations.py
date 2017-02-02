import numpy as np

def func(x): return np.sin(x/5.) * np.exp(x/10.) + 5. * np.exp(-x/2.)

def buildEquation(points, size):
    matrix = []
    vector = []
    for point in points:
        vector.append(func(point))
        line = [point**pow for pow in xrange(0, size+1)]
        matrix.append(line)
    return (np.array(matrix), np.array(vector))

def buildResult(solved):
    return ' '.join([str(val) for val in solved])

result = ''
for input in [([1.,15.], 1), ([1.,8.,15.],2), ([1.,4.,10.,15.],3)]:
    a,b = buildEquation(input[0], input[1])
    result = buildResult(np.linalg.solve(a, b))
    print result
    
print 'Result: ', result

submissionFile = open('..\Results\submission-2.txt', 'w')
submissionFile.write(result)
submissionFile.close()
