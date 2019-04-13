import sys
import tensorflow as tf
sys.path.append('..')
from utils import *


class BowlingNNet:
    def __init__(self, args):
        self.args = args
        # Renaming functions
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        Sigmoid = tf.nn.sigmoid
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense

        # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder(tf.float32, shape=[None, args.input_size])
            self.dropout = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool, name="is_training")
            input1 = Dense(self.input, args.input_size, Relu)
            hidden1 = Dense(input1, 2*args.input_size, Relu)
            hidden2 = Dropout(Dense(hidden1, 4*args.input_size, Relu), .4)
            hidden3 = Dropout(Dense(hidden2, 2*args.input_size, Relu), .2)
            self.bowler = Dense(hidden3, args.bowler_action_size, Sigmoid)

            # self.calculate_loss()

    def calculate_loss(self):
        self.target_action = tf.placeholder(tf.float32, shape=[None, self.args.bowler_action_size])
        self.loss = tf.losses.softmax_cross_entropy(self.target_action, self.bowler)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss)
