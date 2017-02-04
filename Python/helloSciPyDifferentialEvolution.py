import numpy as np
import scipy.optimize as opt
from matplotlib import pylab as plt

def func(x): return np.sin(x/5.) * np.exp(x/10.) + 5. * np.exp(-x/2.) #[1;30]

pointsx = np.arange(1,31)
pointsy = func(pointsx)
plt.plot(pointsx, pointsy)

result = opt.differential_evolution(func, [(1,30)])
print result
print round(result.fun[0], 2)

resultPath = '..\Results\submission-2.txt'

submissionFile = open(resultPath, 'w')
submissionFile.write(str(round(result.fun[0], 2)))
submissionFile.close()