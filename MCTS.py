import math
import numpy as np
import random 
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        
        for i in range(self.args.numMCTSSims):
            self.searchBatsman(state)

        s = self.game.stringRepresentation(state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestA = np.argmax(counts)
            shot_probability = [0]*len(counts)
            shot_probability[bestA] = 1
        else:
            counts = [x**(1./temp) for x in counts]
            shot_probability = [x/float(sum(counts)) for x in counts]

        # calculating which bowler's probability
        counts = [0]*self.game.getActionSize()
        for i in range(self.game.getActionSize()):
            counts[i] = sum([self.Nsa[((s, a), i)] if ((s, a), i) in self.Nsa else 0 for a in range(self.game.getActionSize())])

        if temp == 0:
            bestA = np.argmax(counts)
            bowler_probability = [0] * len(counts)
            bowler_probability[bestA] = 1
        else:
            counts = [x ** (1. / temp) for x in counts]
            bowler_probability = [x / float(sum(counts)) for x in counts]

        # TODO: Probability distribution should be more diverse
        return shot_probability, bowler_probability

    def searchBatsman(self, state):
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
            v: the negative of the value of the current state
        """

        s = self.game.stringRepresentation(state)

        # selecting from the previous generated data if already visited
        # if s is in win state
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(state, "batsman")
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        # this is leaf node and hence value is returned as per the neural network
        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predictshot(state)
            assert len(self.Ps[s]) == 5
            valids = [1] * self.game.getBatsmanActionSize()
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
                print(self.Ps[s], "uniform distribution line 112 MCTS.py")
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                                1 + self.Nsa[(s, a)])
                    # u = self.Qsa[(s,a)] + math.sqrt(2*self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct * (self.Ps[s][a]) * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                    # u = math.sqrt(self.Ns[s] + EPS) 

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        # next_s = self.game.getNextState(state, a)
        assert a < 5
        v = self.searchBowler(state, a)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1

        return -v

    def searchBowler(self, state, shot):
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
            v: the negative of the value of the current state
        """

        s = self.game.stringRepresentation(state)

        # selecting from the previous generated data if already visited
        # if s is in win state
        if (s, shot) not in self.Es:
            self.Es[(s, shot)] = self.game.getGameEnded(state, "bowler")
        if self.Es[(s, shot)] != 0:
            # terminal node
            return -self.Es[(s, shot)]

        # this is leaf node and hence value is returned as per the neural network
        if (s, shot) not in self.Ps:
            # leaf node
            self.Ps[(s, shot)], v = self.nnet.predictBowler(state)
            if()
            valids = self.game.getBowlerValidMoves(state)
            if np.sum(valids) ==0:
                print("ola",state)
            self.Ps[(s, shot)] = self.Ps[(s, shot)] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[(s, shot)])
            if sum_Ps_s > 0:
                self.Ps[(s, shot)] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                print("probab is ", self.Ps[(s, shot)], "sum probab is ", np.sum(self.Ps[(s, shot)]),"state is ", s,
                      "action ", valids)
                self.Ps[(s, shot)] = self.Ps[(s, shot)] + valids
                self.Ps[(s, shot)] /= np.sum(self.Ps[(s, shot)])
                print(self.Ps[(s, shot)], "uniform distribution line 200 MCTS.py")
            self.Vs[(s, shot)] = valids
            self.Ns[(s, shot)] = 0
            return -v

        valids = self.Vs[(s, shot)]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if ((s, shot), a) in self.Qsa:
                    u = self.Qsa[((s, shot), a)] + self.args.cpuct * self.Ps[(s, shot)][a] * math.sqrt(self.Ns[(s, shot)]) / (
                                1 + self.Nsa[((s, shot), a)])
                    # u = self.Qsa[(s,a)] + math.sqrt(2*self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct * self.Ps[(s, shot)][a] * math.sqrt(self.Ns[(s, shot)] + EPS)  # Q = 0 ?
                    # u = math.sqrt(self.Ns[s] + EPS)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        assert a < 5    # bowler should be less than 5
        next_s = self.game.getNextState(state, a, shot)

        v = self.searchBatsman(next_s)

        if ((s, shot), a) in self.Qsa:
            self.Qsa[((s, shot), a)] = (self.Nsa[((s, shot), a)] * self.Qsa[((s, shot), a)] + v) / (self.Nsa[((s, shot), a)] + 1)
            self.Nsa[((s, shot), a)] += 1

        else:
            self.Qsa[((s, shot), a)] = v
            self.Nsa[((s, shot), a)] = 1

        self.Ns[(s, shot)] += 1

        return -v
