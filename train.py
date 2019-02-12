import tensorflow as tf
import numpy as np
from model import SOM
from util import analyze
##################
# set parameters #
##################

trial = "100x100"
data_path = "./data/instances.npy"
#log_path = "~/cache/tensorboard-logdir/jan"
log_path = "./data/"
num_steps = 100
x = 50
y = 50
gpu=None
#get the model
som = SOM(trial, x, y, num_steps, data_path, log_path, gpu=gpu, norm=True,
          learning_rate=0.1)

#train the model
som.train()

# get the trained map
#net = som.get_weights()

### save/load weights
som.save_weights()
#som.load_weights("./data/test.npy")

# get the location / cluster of the data
data_map, clstr_map = som.map_data()

########################################################
###########
# ANALYZE #
###########

idxs = np.load("./data/idxs.npy")
code_lbl = np.load("./data/code_lbl.npy")

pred_counts, pred_acc, best_pred = analyze(clstr_map, code_lbl, 0.6, 2)
