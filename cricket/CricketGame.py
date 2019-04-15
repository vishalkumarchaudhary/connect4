import sys
import numpy as np
from Game import Game
sys.path.append('..')

class CricketGame(Game):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, run_score = 0,wicket_in_hand = 10,overs_left = 50,
                 left_overs_bowler1 = 10, left_overs_bowler2=10, left_overs_bowler3 = 10,
                 left_overs_bowler4 = 10, left_overs_bowler5=10):
        Game.__init__(self)
        self.run_score = run_score
        self.wicket_in_hand = wicket_in_hand
        self.overs_left = overs_left
        self.left_overs_bowler1 = left_overs_bowler1
        self.left_overs_bowler2 = left_overs_bowler2
        self.left_overs_bowler3 = left_overs_bowler3
        self.left_overs_bowler4 = left_overs_bowler4
        self.left_overs_bowler5 = left_overs_bowler5
        self.bowler_action_size = 5
        self.batsman_action_size = 5
        self.state = np.asarray([self.run_score, self.wicket_in_hand, self.overs_left,self.left_overs_bowler1, self.left_overs_bowler2, self.left_overs_bowler3,self.left_overs_bowler4, self.left_overs_bowler5])

    def getInitBoard(self):
        return self.state

    def getBoardSize(self):
        return np.shape([self.state])

    def getCanonicalForm(self, board, player):
        pass

    def getActionSize(self):
        # TODO: Change this function name and corresponding things
        return self.batsman_action_size

    def getBatsmanActionSize(self):
        return self.batsman_action_size

    def getBowlerValidMoves(self, state):
        return [state[-5] > 0, state[-4] > 0, state[-3] > 0, state[-2] > 0, state[-1] > 0]

    def getBowlerActionSize(self):
        return self.bowler_action_size

    def stringRepresentation(self, state):
        return str(state)

    def getNextState(self, state, bowler, shot):
        """Returns a copy of the board with updated move, original board is unmodified."""
        batsman = int(state[1]-9)
        i = batsman
        score = [1,2,3,4,6]
        pw_min = np.zeros((5,5))
        pw_max = np.zeros((5,5))
        pw_min[0] = [0.01,0.02,0.03,0.1,0.3]
        pw_min[1] = [0.007,0.012,0.021,0.07,0.15]
        pw_min[2] = [0.005,0.01,0.015,0.05,0.1]
        pw_min[3] = [0.002,0.004,0.006,0.02,0.07]
        pw_min[4] = [0.001,0.002,0.003,0.01,0.05]
        pw_max[0] = [0.1,0.2,0.3,0.5,0.7]
        pw_max[1] = [0.07,0.12,0.21,0.4,0.6]
        pw_max[2] = [0.05,0.1,0.015,0.3,0.5]
        pw_max[3] = [0.02,0.04,0.006,0.2,0.4]
        pw_max[4] = [0.01,0.02,0.03,0.1,0.3]

        p_maxfactor = [0.7,0.75,0.85,0.9,0.95]
        p_minfactor = [0.3,0.35,0.4,0.45,0.5]
        wicket = 0
        runs = 0
        p_w = pw_min[bowler][shot]+(pw_max[bowler][shot]-pw_min[bowler][shot])*(i-1)/9
        p_factor = p_maxfactor[bowler]+(p_minfactor[bowler]-p_maxfactor[bowler])*(i-1)/9

        for balls in range(0, 6):
            if i < 11:
                sample = np.random.rand(1)
                if sample < p_w:
                    wicket = wicket+1
                    i = i+1
                    p_w = pw_min[bowler][shot]+(pw_max[bowler][shot]-pw_min[bowler][shot])*(i-1)/9
                    p_factor = p_maxfactor[bowler]+(p_minfactor[bowler]-p_maxfactor[bowler])*(i-1)/9
                else:
                    runs = runs+p_factor*score[shot]
        bowlers_over = np.zeros((5,))
        bowlers_over[bowler] = 1
        return np.asarray([state[0] + runs, state[1] - wicket, state[2]-1, state[3]-bowlers_over[0],
                state[4]-bowlers_over[1], state[5]-bowlers_over[2], state[6]-bowlers_over[3],
                state[7]-bowlers_over[4]])

        # self.overs_left -= 1
        # self.run_score += runs
        # self.wicket_in_hand -= wicket
        # self.left_overs_bowler1 -= bowlers_over[0]
        # self.left_overs_bowler2 -= bowlers_over[1]
        # self.left_overs_bowler3 -= bowlers_over[2]
        # self.left_overs_bowler4 -= bowlers_over[3]
        # self.left_overs_bowler5 -= bowlers_over[4]

        # return np.asarray([self.run_score, self.wicket_in_hand, self.overs_left,
        #                    self.left_overs_bowler1, self.left_overs_bowler2, self.left_overs_bowler3,
        #                    self.left_overs_bowler4, self.left_overs_bowler5])

    def getReward(self, batsman, bowler, shot):
        i = batsman
        score = [1, 2, 3, 4, 6]
        pw_min = np.zeros((5, 5))
        pw_max = np.zeros((5, 5))
        pw_min[0] = [0.01, 0.02, 0.03, 0.1, 0.3]
        pw_min[1] = [0.007, 0.012, 0.021, 0.07, 0.15]
        pw_min[2] = [0.005, 0.01, 0.015, 0.05, 0.1]
        pw_min[3] = [0.002, 0.004, 0.006, 0.02, 0.07]
        pw_min[4] = [0.001, 0.002, 0.003, 0.01, 0.05]
        pw_max[0] = [0.1, 0.2, 0.3, 0.5, 0.7]
        pw_max[1] = [0.07, 0.12, 0.21, 0.4, 0.6]
        pw_max[2] = [0.05, 0.1, 0.015, 0.3, 0.5]
        pw_max[3] = [0.02, 0.04, 0.006, 0.2, 0.4]
        pw_max[4] = [0.01, 0.02, 0.03, 0.1, 0.3]

        p_maxfactor = [0.7, 0.75, 0.85, 0.9, 0.95]
        p_minfactor = [0.3, 0.35, 0.4, 0.45, 0.5]
        wicket = 0
        runs = 0
        p_w = pw_min[bowler][shot] + (pw_max[bowler][shot] - pw_min[bowler][shot]) * (i - 1) / 9
        p_factor = p_maxfactor[bowler] + (p_minfactor[bowler] - p_maxfactor[bowler]) * (i - 1) / 9

        for balls in range(0, 6):
            if i < 11:
                sample = np.random.rand(1)
                if sample < p_w:
                    wicket = wicket + 1
                    i = i + 1
                    p_w = pw_min[bowler][shot] + (pw_max[bowler][shot] - pw_min[bowler][shot]) * (i - 1) / 9
                    p_factor = p_maxfactor[bowler] + (p_minfactor[bowler] - p_maxfactor[bowler]) * (i - 1) / 9
                else:
                    runs = runs + p_factor * score[shot]
        return runs, wicket

    def getValidMoves(self, state):
        return [1]*self.getActionSize()

    # TODO: fix the reward function for both batsman and bowler
    def getGameEnded(self, state, player="batsman"):
        if state is not None:
            if state[1] <= 0 or state[2] <= 0:#or state[3] <= 0 or state[4] <= 0 or state[5] <= 0 or state[6] <= 0 or state[7] <= 0:
                if player == 'batsman':
                    return state[0]
                elif player == 'bowler':
                    return -state[0]
                else:
                    raise ValueError('Unexpected winstate found: ')
            else:
                # 0 used to represent unfinished game.
                return 0
        return 0

    def display(self):
        print(" -----------------------")
        print("Runs ", self.run_score, " wickets ", 10 - self.wicket_in_hand)
        print(" -----------------------")