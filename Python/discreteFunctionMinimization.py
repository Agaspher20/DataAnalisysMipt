import numpy as np
import scipy.optimize as opt
from matplotlib import pylab as plt

def func(x): return np.sin(x/5.) * np.exp(x/10.) + 5. * np.exp(-x/2.) #[1;30]
def h(x): return int(func(x)) #[1;30]

pointsx = np.arange(0,31,0.5)
pointsy = [h(x) for x in pointsx]
plt.plot(pointsx, pointsy)

minimumBfgs = opt.minimize(h,30,method='BFGS')
minimumEvo = opt.differential_evolution(h, [(1,30)])
result = str(round(minimumBfgs.fun,2)) + ' ' + str(round(minimumEvo.fun, 2))

print minimumBfgs
print minimumEvo
print result

resultPath = '..\Results\submission-3.txt'

submissionFile = open(resultPath, 'w')
submissionFile.write(result)
submissionFile.close()
