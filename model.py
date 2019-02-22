import tensorflow as tf
import numpy as np
from util import placeholder, normalize, pipe, profile
from tqdm import tqdm
import os

class SOM(object):
    def __init__(self, trial, x, y, num_steps, data_path, log_path,
                 learning_rate=0.5, norm=False, radius=None, tensorboard=False, gpu=None):
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
        self.log_path = log_path
        self.x = x
        self.y = y
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.radius = radius
        self.norm = norm
        self.tensorboard = tensorboard
        self.gpu = gpu

        if gpu != None:
            ##########
            ### PICK GPU ###################
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        with tf.variable_scope("input"):
            with tf.variable_scope("data"):
                if norm is True:
                    self.data = normalize( np.load(data_path).astype(np.float32))
                else:
                    self.data = np.load(data_path).astype(np.float32)

                self.feat_nr = len(self.data[0])
                self.nr_inst = len(self.data)
            with tf.variable_scope("pipeline"):
                inpt = pipe(self.data, prefetch=20)

        # step count
        self.step = tf.placeholder(tf.float32)

        # create weight matrix
        self.weights = tf.get_variable("weights", shape=[self.x*self.y, self.feat_nr],
                                       initializer=tf.random_uniform_initializer)

                                       #initializer=tf.random_normal_initializer)

        with tf.variable_scope("locations"):
            self.locations = tf.constant(np.array(list([np.array([i, j]) for i in range(self.x) for j in range(self.y)])))

        if radius == None:
            self.radius = tf.constant(max(self.x, self.y)/2, dtype=tf.float32)

        #Calculate current learning rate and radius
        # todo: different decay options
        with tf.variable_scope("decay"):
            decay = tf.cast(1 - (self.step/self.num_steps), dtype=tf.float32)

        with tf.variable_scope("current_lr"):
            current_lr = self.learning_rate * decay
        with tf.variable_scope("current_radius"):
            current_radius = self.radius * decay


        with tf.variable_scope("bmu"):
            # calculate Best Matching Unit (euclidean distances)
            distances = tf.sqrt(tf.reduce_sum((self.weights - inpt)**2, 1))
            bmu = tf.reduce_min(distances)
            #bmu = tf.argmin(distances, 0)

        with tf.variable_scope("bmu_loc"):
            # get the location of the bmu
            #mask = tf.pad(tf.reshape(bmu, [1]), tf.constant(np.array([[0, 1]])))
            #size = tf.cast(tf.constant(np.array([1, 2])), dtype=tf.int64)
            #bmu_loc = tf.reshape(tf.slice(self.locations, mask, size), [2])
            mask = tf.equal(distances, bmu)
            bmu_locs = tf.boolean_mask(self.locations, mask)
            bmu_loc = tf.slice(bmu_locs, [0, 0], [1, 2])[0] # make sure its only one location

        with tf.variable_scope("neighborhood"):
            # calculate the influence on the neighborhood of the bmu
            bmu_dist = tf.sqrt(tf.cast(tf.reduce_sum((self.locations - bmu_loc)**2, 1), dtype=tf.float32))
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
            wrtr = tf.summary.FileWriter(os.path.expanduser(self.log_path+self.trial))
            wrtr.add_graph(self.sess.graph)
            profile(self.sess, wrtr, new_weights, feed_dict= None, prerun=3, tag='flow')
            wrtr.flush()

    def train(self, save_point=None):
        """train the SOM model
           save_point: int, save model every x steps"""
        for s in tqdm(range(self.num_steps)):
            for instance in range(self.nr_inst):
                self.sess.run(self.train_step, feed_dict={self.step: s})
            if (save_point != None) and (s % save_point == 0):
                self.save_weights(self.log_path + self.trial+str(s))
        print("Training done")

    def get_weights(self):
        """get weights in x*y-matrix form"""
        return tf.reshape(self.weights, (self.x, self.y, self.feat_nr)).eval()

    def save_weights(self, path=None):
        """saves the trained weights in the folder of path under the name of the trial"""
        if path == None:
            path = self.log_path + self.trial
        np.save(path, self.weights.eval())
        print("weights were saved under:", path)

    def load_weights(self, path):
        try:
            self.sess.run(self.weights.assign(np.load(path)))
        except ValueError:
            data = np.load(path)
            shape = data.shape
            self.sess.run(self.weights.assign(np.reshape(data,(shape[0]*shape[1], shape[2]) )))
        print("weights were loaded")

    def map_data(self, data=None):
        """ maps the given data to the weigth matrix, according to similarity
            also maps the given data to clusters by numbers

            returns loc_map, cluster_map"""
        weights = self.weights.eval()
        locations = self.locations.eval()
        if data == None:
            data = self.data
        loc_map, clstr_map, loc2id, id2loc, count = [], [], {}, {}, 0
        for i in tqdm(range(len(data))):
            # get the location
            min_idx = min([(np.linalg.norm(data[i]-weight), idx)
                           for idx,weight in enumerate(weights)])[1]
            loc_map.append(locations[min_idx])

            # get cluster number
            loc_str= str(locations[min_idx])
            if not loc_str in loc2id:
                loc2id[loc_str] = count
                id2loc[count] = loc_str
                count +=1
            clstr_map.append(loc2id[loc_str])
        return loc_map, clstr_map
