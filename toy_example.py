import tensorflow as tf
import numpy as np
from model import SOM

def plot_colour(net):
    from matplotlib import pyplot as plt
    from matplotlib import patches as patches
    fig = plt.figure()
    # setup axes
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, net.shape[0]+1))
    ax.set_ylim((0, net.shape[1]+1))
    ax.set_title('Self-Organising Map after %d iterations' % num_steps)

    # plot the rectangles
    for x in range(1, net.shape[0] + 1):
        for y in range(1, net.shape[1] + 1):
            ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                                           facecolor=net[x-1,y-1,:],
                                           edgecolor='none'))
    plt.show()

# Toyset
feat_nr = 3
data = np.random.randint(0, 255, (100,3))
np.save("./data/toydata.npy", data)
data_path = "./data/toydata.npy"



##################
# set parameters #
##################

trial = "toytest"
log_path = "./data/"
num_steps = 100
x = 10
y = 10
gpu=None
#get the model
som = SOM(trial, x, y, num_steps, data_path, log_path, gpu=gpu, norm=True,
          learning_rate=0.1)

#train the model
som.train()

# get the trained map
net = som.get_weights()

plot_colour(net)
