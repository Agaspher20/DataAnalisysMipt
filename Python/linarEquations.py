import numpy as np

def func(x):
    return np.sin(x/5.) * np.exp(x/10.) + 5. * np.exp(-x/2.)

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

a1,b1 = buildEquation([1.,15.], 1)
print buildResult(np.linalg.solve(a1, b1))

a2,b2 = buildEquation([1.,8.,15.],2)
print buildResult(np.linalg.solve(a2, b2))

a3,b3 = buildEquation([1.,4.,10.,15.],3)
solved = buildResult(np.linalg.solve(a3, b3))
print solved
print 'Result: ', solved

submissionFile = open('..\Results\submission-2.txt', 'w')
submissionFile.write(solved)
submissionFile.close()
