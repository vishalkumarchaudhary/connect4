connect4 folder is game related program file
	
	Connect4Game.py :
		it defines small function about the games like getting board states and action states , ending game function
	
	Connect4Logic.py :
		It defines function for setting board , valid moves , how to move with action , function for state be in win.

	Connect4Players.py :
		This defines class each containing play function which decides how players is going to chose action. Like the players defines here are randomplayer , human player and onestep-lookahead-player



------------------------------------------------------------------
MCTS.py :
	This file uses monte carlo tree search for generating good policy during the self play.

Coach.py :
	this contain the learning method of neural network 

neural network is defined inside connect4/tensorflow folder.

Arena.py :
	This contains method which will do the tournament between the agents.