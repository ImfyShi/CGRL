import numpy as np
from MCTS import MCTS

class NNtBasedMCTSPlayer():
    def __init__(self, game, nnt, args, temp=0, percentile=0.68):
        self.game=game
        self.nnt=nnt
        self.args=args
        self.temp=temp
        self.percentile=percentile

    def play(self, board):
        mcts=MCTS(self.game, self.nnt, self.args, self.percentile)
        prob, validmoves=mcts.getActionProb(board, self.temp)
        action=np.random.choice(len(prob), p=prob)
        return validmoves[action]
