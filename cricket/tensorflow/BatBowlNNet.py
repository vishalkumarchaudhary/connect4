import sys
import tensorflow as tf
sys.path.append('..')
from utils import *


class BatBowlNN:
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
            hidden21 = Dropout(Dense(hidden1, 4*args.input_size, Relu), .2)
            hidden22 = Dropout(Dense(hidden1, 4 * args.input_size, Relu), .2)
            hidden31 = Dropout(Dense(hidden21, 2*args.input_size, Relu), .2)
            hidden32 = Dropout(Dense(hidden22, 2 * args.input_size, Relu), .2)
            self.shot = Dense(hidden31, args.batsman_action_size, Sigmoid)
            self.bowler = Dense(hidden32, args.batsman_action_size, Sigmoid)

            self.calculate_loss()

    def calculate_loss(self):

        self.target_shot = tf.placeholder(tf.float32, shape=[None, self.args.batsman_action_size])
        self.target_bowler = tf.placeholder(tf.float32, shape=[None, self.args.batsman_action_size])

        self.batting_loss = tf.losses.softmax_cross_entropy(self.target_shot, self.shot)
        self.bowling_loss = tf.losses.softmax_cross_entropy(self.target_bowler, self.bowler)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.batting_train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.batting_loss)
            self.bowling_train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.bowling_loss)

