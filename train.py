import tensorflow as tf
import numpy as np
from model import SOM

# Toyset
#feat_nr = 3
#num_iter = 100
#data = normalize(np.random.randint(0, 255, (100,3)))
#np.save("./data/test.npy", data)
#data_path = "./data/test.npy"


##################
# set parameters #
##################
data_path = "./data/instances.npy"
num_steps = 100
x = 5
y = 5

#get the model
som = SOM(data_path, x, y, num_steps)

#train the model
som.train()

# get the trained map
net = som.get_weights()


#from matplotlib import pyplot as plt
#from matplotlib import patches as patches
#fig = plt.figure()
# setup axes
#ax = fig.add_subplot(111, aspect='equal')
#ax.set_xlim((0, net.shape[0]+1))
#ax.set_ylim((0, net.shape[1]+1))
#ax.set_title('Self-Organising Map after %d iterations' % num_iter)

# plot the rectangles
#for x in range(1, net.shape[0] + 1):
#    for y in range(1, net.shape[1] + 1):
#        ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
#                     facecolor=net[x-1,y-1,:],
#                     edgecolor='none'))
#plt.show()
