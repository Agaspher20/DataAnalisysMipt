#%%
from statsmodels.stats.proportion import proportion_confint
import numpy as np

n=50
n_success=1
alfa=0.05

#%%
normal_interval = proportion_confint(n_success, n, method = 'normal')
print np.round(normal_interval, 4)

#%%
normal_interval = proportion_confint(n_success, n, method = 'wilson')
print np.round(normal_interval, 4)

#%%
from statsmodels.stats.proportion import samplesize_confint_proportion

n_samples = int(np.ceil(samplesize_confint_proportion(0.02, 0.01)))

print n_samples

#%%
samples = [(0.01*x,int(np.ceil(samplesize_confint_proportion(0.01*x, 0.01)))) for x in xrange(0,100,1)]

#%%
%matplotlib inline
import matplotlib.pyplot as plt

samples_x,samples_y = zip(*samples)
line_train = plt.plot(samples_x, samples_y)

#%%
print max(samples, key=lambda (x,y): y)
