import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        state = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(state) == 0 and np.sum(self.game.getBowlerValidMoves(state)) > 0:
            it += 1
            if verbose:
                assert (self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(state)
            shot_p, bowler_p = players[curPlayer + 1](state)

            valids = self.game.getBowlerValidMoves(state)

            if valids[np.argmax(bowler_p)] == 0:
                print("From 51 of Arena.py", bowler_p, state, valids)
                break
            #     assert valids[action] > 0
            state = self.game.getNextState(state, np.argmax(bowler_p), np.argmax(shot_p))
        if verbose:
            assert (self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(state, 1)))
            self.display(state)
        run1 = self.game.getGameEnded(state)
        print("First network successfully played")
        curPlayer = -1
        state = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(state) == 0 and np.sum(self.game.getBowlerValidMoves(state)) > 0:
            it += 1
            if verbose:
                assert (self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(state)
            shot_p, bowler_p = players[curPlayer + 1](state)

            valids = self.game.getBowlerValidMoves(state)

            if valids[np.argmax(bowler_p)] == 0:
                print("From 51 of Arena.py", bowler_p, state, valids)
                break
            #     assert valids[action] > 0
            state = self.game.getNextState(state, np.argmax(bowler_p), np.argmax(shot_p))
        if verbose:
            assert (self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(state, 1)))
            self.display(state)
        run2 = self.game.getGameEnded(state)
        return run1 > run2


    def playGames(self, num, verbose=False):
            """
            Plays num games in which player1 starts num/2 games and player2 starts
            num/2 games.

            Returns:
                oneWon: games won by player1
                twoWon: games won by player2
                draws:  games won by nobody
            """
            eps_time = AverageMeter()
            bar = Bar('Arena.playGames', max=num)
            end = time.time()
            eps = 0
            maxeps = int(num)

            num = int(num/2)
            oneWon = 0
            twoWon = 0
            draws = 0
            for _ in range(num):
                gameResult = self.playGame(verbose=verbose)
                if gameResult is True:
                    oneWon+=1
                elif gameResult is False:
                    twoWon+=1
                else:
                    draws+=1
                # bookkeeping + plot progress
                eps += 1
                eps_time.update(time.time() - end)
                end = time.time()
                bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=maxeps, et=eps_time.avg,
                                                                                                           total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()

            self.player1, self.player2 = self.player2, self.player1

            for _ in range(num):
                gameResult = self.playGame(verbose=verbose)
                if gameResult is False:
                    oneWon+=1
                elif gameResult is True:
                    twoWon+=1
                else:
                    draws+=1
                # bookkeeping + plot progress
                eps += 1
                eps_time.update(time.time() - end)
                end = time.time()
                bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=num, et=eps_time.avg,
                                                                                                           total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()

            bar.finish()

            return oneWon, twoWon, draws
