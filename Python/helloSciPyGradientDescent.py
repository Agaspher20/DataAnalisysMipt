import numpy as np
import scipy.optimize as opt
from matplotlib import pylab as plt

def func(x): return np.sin(x/5.) * np.exp(x/10.) + 5. * np.exp(-x/2.) #[1;30]

pointsx = np.arange(0,31)
pointsy = func(pointsx)
plt.plot(pointsx, pointsy)

minimums = [opt.minimize(func,point,method='BFGS') for point in [2,30]]
result = ' '.join([str(round(min.fun, 2)) for min in minimums])
print result

resultPath = '..\Results\submission-1.txt'

submissionFile = open(resultPath, 'w')
submissionFile.write(result)
submissionFile.close()
