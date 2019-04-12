from collections import namedtuple
import numpy as np

class State():

    def __init__(self, run_score=0,wicket_in_hand=10,overs_left
                 left_overs_bowler1=10, left_overs_bowler2=10, left_overs_bowler3=10, 
                 left_overs_bowler4=10, left_overs_bowler5=10):
        self.run_score = run_score
        self.wicket_in_hand = wicket_in_hand
        self.overs_left = overs_left
        self.left_overs_bowler1 = left_overs_bowler1
        self.left_overs_bowler2 = left_overs_bowler2
        self.left_overs_bowler3 = left_overs_bowler3
        self.left_overs_bowler4 = left_overs_bowler4
        self.left_overs_bowler5 = left_overs_bowler5
        self.left_overs_bowler6 = left_overs_bowler6
        self.bowler_action_size = 5
        self.batsman_action_size = 5
        self.state = np.asarray([self.run_score , self.wicket_in_hand,self.overs_left,
                                    left_overs_bowler1,left_overs_bowler2, left_overs_bowler3, 
                                    left_overs_bowler4 ,left_overs_bowler5])

    # def get_valid_moves(self):
    #     return [1]*8
