# %%writefile main.py
from Coach import Coach
from connect4.CricketGame import CricketGame
from connect4.tensorflow.NNet import NNetWrapper as nn
from utils import dotdict
import time
import numpy as np
# from utils import *
print(time.time())
args = dotdict({
    'numIters': 1000,
    'numEps': 10,
    'tempThreshold': 15,
    'updateThreshold': 0.5,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 100,
    'arenaCompare': 10,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    # 'load_folder_file': ('temp','best.pth.tar'),
    'load_folder_file': ('temp/', "checkpoint_0.pth.tar"),
    
    'numItersForTrainExamplesHistory': 20,

})

if __name__ == "__main__":
    
    # game instantiating the game connect4 game
    g = CricketGame()
    # initialising neural network and the config of neural network is defined in connect4/tensorflow/NNet.py file
    nnet = nn(g)
    print("**********************")
    print(args.load_folder_file[0], args.load_folder_file[1])
    print("_______________________________________________________")
    x = [[], [], []]
    np.save("losses_array.npy", x)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    # Coach helps to generate training examples and the learn from the examples of generated during self-play
    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    print(time.asctime())
    c.learn()
