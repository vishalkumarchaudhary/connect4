import os
import shutil
import time
import random
import numpy as np
import sys
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet

import tensorflow as tf
from .BatBowlNNet import BatBowlNN as net

sys.path.append('../../')

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 2,
    'batch_size': 4,
    'num_channels': 512,
    'input_size': 8,
    'bowler_action_size': 5,
    'batsman_action_size': 5


})


# Code based on othello.NNetWrapper with minimal changes.

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.game = game
        self.NNet = net(args)

        self.sess = tf.Session(graph=self.NNet.graph)
        self.saver = None
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        self.sess.run(tf.variables_initializer(self.NNet.graph.get_collection('variables')))

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        losses = [[], []]
        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            data_time = AverageMeter()
            batch_time = AverageMeter()
            batting_action_loss = AverageMeter()
            bowling_action_loss = AverageMeter()
            end = time.time()
            
            bar = Bar('Training Net', max=int(len(examples) / args.batch_size))
            batch_idx = 0

            # self.sess.run(tf.local_variables_initializer())
            while batch_idx < int(len(examples) / args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                states, batting_action, bowling_action = list(zip(*[examples[i] for i in sample_ids]))

                # predict and compute gradient and do SGD step
                _dict = {self.NNet.input: states, self.NNet.target_shot: batting_action,
                         self.NNet.isTraining: True, self.NNet.target_bowler: bowling_action}

                # measure data loading time
                data_time.update(time.time() - end)

                # record loss
                self.sess.run([self.NNet.shot,self.NNet.bowler], feed_dict=_dict)
                _, _, batting_loss, bowling_loss = self.sess.run([self.NNet.batting_train_step, self.NNet.bowling_train_step,
                                                                         self.NNet.batting_loss, self.NNet.bowling_loss], feed_dict=_dict)

                batting_action_loss.update(batting_loss, len(states))
                bowling_action_loss.update(bowling_loss, len(states))

                losses[0].append(batting_loss)
                losses[1].append(bowling_loss)
                batch_idx += 1
                # measure elapsed time
                batch_time.update(time.time() - end)
                
                bar.next()
            bar.finish()
        return losses

    def predictshot(self, state):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        state = state[np.newaxis, :]

        # # run
        # shots, bowler = self.sess.run([self.battingNNet.shot, self.bowlingNNet.bowler],
        #                               feed_dict={self.battingNNet.input: state,
        #                                          self.bowlingNNet.input: state})
        shots = self.sess.run(self.NNet.shot, feed_dict={self.NNet.input: state})
        bowler = self.sess.run(self.NNet.bowler, feed_dict={self.NNet.input: state})

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        runs, _ = self.game.getReward(self.game.wicket_in_hand-9,np.argmax(bowler), np.argmax(shots))
        if (np.sum(shots) == 0):
            print("batting probab in NNet.py", shots[0], "State is ", state)
        return shots[0], runs

    def predictBowler(self, state):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        state = state[np.newaxis, :]

        # # run
        # shots, bowler = self.sess.run([self.battingNNet.shot, self.bowlingNNet.bowler],
        #                               feed_dict={self.battingNNet.input: state,
        #                                          self.bowlingNNet.input: state})
        shots = self.sess.run(self.NNet.shot, feed_dict={self.NNet.input: state})
        bowler = self.sess.run(self.NNet.bowler, feed_dict={self.NNet.input: state})
        # print("action space of MCTS is ", len((shots.T * bowler).flatten()))
        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time() - start))
        runs, _ = self.game.getReward(self.game.wicket_in_hand - 9, np.argmax(bowler), np.argmax(shots))
        if(np.sum(bowler)==0):
            print("bowler probab in NNet.py", bowler[0],"State is ",state)
        return bowler[0], runs

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")

        if self.saver == None:
            self.saver = tf.train.Saver(self.NNet.graph.get_collection('variables'))
        with self.NNet.graph.as_default():
            self.saver.save(self.sess, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath+'batting' + '.meta'):
            raise("No model in path {}".format(filepath))
        # TODO: pass the neural network as argument
        with self.NNet.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, filepath)

        # with self.nnet.graph.as_default():
        #     self.saver = tf.train.Saver()
        #     self.saver.restore(self.sess, filepath)
