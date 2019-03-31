import math
import numpy as np
import random 
from math import log
from numpy.random import beta
EPS = 1e-6
DELTA = 1e-5

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        
        self.Beta = {} 
        self.QMeanSA = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s
        

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        
        for i in range(self.args.numMCTSSims):
            # import ipdb; ipdb.set_trace()   # for debugging
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs
    
    def updateQ(self,reward ,state ,action):
        # updating Qmean of the action
        s = state
        r = reward
        a = action
        if (s,a) in self.QMeanSA:
                self.QMeanSA[(s,a)] = (self.Nsa[(s,a)]*self.QMeanSA[(s,a)] + r)/(self.Nsa[(s,a)]+1)
                self.Nsa[(s,a)] += 1
        else:
            self.QMeanSA[(s,a)] = r
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1

        # calculating q values 
        
        for a in range(self.game.getActionSize()):
            if self.Vs[s][a]:
                self.Beta[(s,a)] = beta(1+self.Nsa[(s,a)]*self.QMeanSA[(s,a)] , 1.000000+self.Nsa[(s,a)] -self.Nsa[(s,a)]*self.QMeanSA[(s,a)] )

    def bestAction(self,s):
        """
        This function choses the maxq value for the state passed
        
        Return :
            action : index of the action to chose
        """

        bestQ = -1e+10
        bestA  = -1

        for a in range(self.game.getActionSize()):

            if self.Vs[s][a] :
                # print(self.Beta[(s,a)],"entered",bestQ,bestQ < self.Beta[(s,a)] )
                if bestQ < self.Beta[(s,a)] :
                    bestA = a 
                    bestQ = self.Beta[(s,a)]


        return bestA

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        # selecting from the previous generated data if already visited
         

        # if s is in win state
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s]!=-1                               :
            # terminal node
            return  1 - self.Es[s]

        # this is leaf node and hence value is returned as per the neural network
        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            v = v[0]
            if(v>=1):
                v = 1
            if v < 0 :
                v = 0 
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
                print(self.Ps[s],"uniform distribution line 100 MCTS.py")
            self.Vs[s] = valids
            for a in range(len(valids)) :
                if valids[a]:
                    self.Beta[(s,a)] = 0
                    self.QMeanSA[(s,a)] = 0
                    self.Nsa[(s,a)] = 0
            self.Ns[s] = 1
            return 1-v

        # # chossing each arm atleat once
        # if self.Ns[s] == 1 :
        #     for a in range(len(self.Vs[s])):
        #         if self.Vs[s][a]:
        #             next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        #             next_s = self.game.getCanonicalForm(next_s, next_player)
        #             v = self.search(next_s)
        #             self.updateQ((1+v)/2,s,a)

        a = self.bestAction(s)
        
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)        
        self.updateQ(1-v ,s,a)
        
        v = 1-v
        return v



