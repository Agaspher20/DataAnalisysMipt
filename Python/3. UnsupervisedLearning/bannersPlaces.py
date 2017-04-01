#%%
import pandas as pd
import numpy as np
data = pd.read_csv('..\..\Data\checkins.csv')
data = data.drop("created_at", 1).drop("id", 1).drop("user_id", 1).drop("venue_id", 1)
data.shape
#%%
import sklearn.cluster as cl
ms = cl.MeanShift(bandwidth=0.1, min_bin_freq=16)
ms.fit(data)

#%%
print "Labels type", type(ms.labels_)
print "Label type", type(ms.labels_[0])
print "Labels count:", len(ms.labels_)
print "Unique labels count:", len(np.unique(ms.labels_))

#%%
offices = [
    [33.751277, -118.188740], #, "(Los Angeles)"],
    [25.867736, -80.324116], #, "(Miami)"],
    [51.503016, -0.075479], #, "(London)"],
    [52.378894, 4.885084], #, "(Amsterdam)"],
    [39.366487, 117.036146], #, "(Beijing)"],
    [-33.868457, 151.205134], #, "(Sydney)"]
]

#%%
def distance(center, office):
    return np.sqrt((center[0]-office[0])**2. + (center[1]-office[1])**2.)

def min_distance(center):
    distances = map(lambda office: (distance(center, office),center,office), offices)
    return min(distances, key=lambda distance: distance[0])
    
centersByDistance = sorted(map(min_distance, ms.cluster_centers_), key=lambda d: d[0])

#%%
result = centersByDistance[0]
print result[1][0], result[1][1]

#%%
def write_answer(latitude, longitude):
    with open("..\..\Results\banners.txt", "w") as fout:
        fout.write(str(latitude) + " " + str(longitude))
write_answer(result[1][0], result[1][1])
