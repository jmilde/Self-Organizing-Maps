import tensorflow as tf
import numpy as np
from util import placeholder, normalize, pipe, profile
from tqdm import tqdm
import os

class SOM(object):
    def __init__(self, trial, data_path, x, y, num_steps, learning_rate=0.5, norm=False, radius=None, tensorboard=True, gpu=None):
        '''
              trial: string, name of the trial
               data: tensorflow dataset iterator
               x, y: int, dimension of the map
            feat_nr: int, number of inpt_features
          num_steps: int, number of training steps
               norm: bool, if True then data will be normalized
             radius: int, if None use default radius max(x,y)/2
        tensorboard: bool, if True, logs the graph and runs a performance test
                gpu: int, picks gpu from cluster by number
        '''

        self.trial = trial
        self.data_path = data_path
        self.x = x
        self.y = y
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.radius = radius
        self.norm = norm
        self.tensorboard = tensorboard
        self.gpu = gpu

        if gpu != None:
            ############# PICK GPU ###################
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        with tf.variable_scope("input"):
            with tf.variable_scope("data"):
                self.data = np.load(data_path)
                if norm is True:
                    self.data = normalize(self.data)
                self.feat_nr = len(self.data[0])
                self.nr_inst = len(self.data)
            with tf.variable_scope("pipeline"):
                pipeline = pipe(self.data, prefetch=10)

            inpt = placeholder(tf.float32, [self.feat_nr], pipeline, "inpt")

        # create weight matrix
        self.weights = tf.get_variable("weights", shape=[self.x*self.y, self.feat_nr],
                                       initializer=tf.random_normal_initializer)

        with tf.variable_scope("locations"):
            locations = tf.constant(np.array(
            list([np.array([i, j]) for i in range(self.x) for j in range(self.y)])))

        if radius == None:
            self.radius = tf.constant(max(self.x, self.y)/2, dtype=tf.float32)

        #Calculate current learning rate and radius
        step = tf.train.get_or_create_global_step()

        with tf.variable_scope("decay"):
            decay = tf.cast(1 - (step/self.num_steps), dtype=tf.float32)

        with tf.variable_scope("current_lr"):
            current_lr = self.learning_rate * decay
        with tf.variable_scope("current_radius"):
            current_radius = self.radius * decay


        with tf.variable_scope("bmu"):
            # calculate Best Matching Unit (euclidean distances)
            distances = tf.sqrt(tf.reduce_sum((self.weights - inpt)**2, 1))
            bmu = tf.argmin(distances, 0)
            #bmu = tf.reduce_min(distances)
            #if len(bmu.shape) > 1:
            #    bmu = bmu[0]

        with tf.variable_scope("bmu_loc"):
            # get the location of the bmu
            mask = tf.pad(tf.reshape(bmu, [1]), tf.constant(np.array([[0, 1]])))
            size = tf.cast(tf.constant(np.array([1, 2])), dtype=tf.int64)
            bmu_loc = tf.reshape(tf.slice(locations, mask, size), [2])
            #mask = tf.equal(distances, bmu)
            #bmu_loc = tf.boolean_mask(locations, mask)


        with tf.variable_scope("neighborhood"):
            # calculate the influence on the neighborhood of the bmu
            bmu_dist = tf.sqrt(tf.cast(tf.reduce_sum((locations - bmu_loc)**2, 1), dtype=tf.float32))
            nbr_func = tf.exp(-bmu_dist / (2 * (current_radius**2)))

        with tf.variable_scope("delta"):
            # get the delta of the weights
            delta = tf.expand_dims(nbr_func, -1) * current_lr * (inpt-self.weights)

        with tf.variable_scope("new_weights"):
            #update the new weights
            new_weights = self.weights + delta
            self.train_step = tf.assign(self.weights, new_weights)

        # initialize session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        if self.tensorboard is True:
            wrtr = tf.summary.FileWriter(os.path.expanduser("~/cache/tensorboard-logdir/jan/"+trial))
            wrtr.add_graph(self.sess.graph)
            profile(self.sess, wrtr, new_weights, feed_dict= None, prerun=3, tag='flow')
            wrtr.flush()


    def train(self):
        """train the SOM model"""
        for iter_nr in tqdm(range(self.num_steps)):
            for instance in range(self.nr_inst):
                self.sess.run(self.train_step)
        print("Training done")

    def get_weights(self):
        """get weights in x*y-matrix form"""
        return tf.reshape(self.weights, (self.x, self.y, self.feat_nr)).eval()

    def save_weights(self, path):
        np.save(path+self.trial, tf.reshape(self.weights, (self.x, self.y, self.feat_nr)).eval())
        print("weights were saved under:", path, trial)
