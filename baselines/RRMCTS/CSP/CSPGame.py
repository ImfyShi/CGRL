from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .CSPLogic import *
import numpy as np
#import matplotlib.pyplot as plt
import copy


class CSPGame(Game):
    def __init__(self, instance_data, maxmum_items_number):
        self.instance_data = instance_data
        self.InitState = self.getInitState(maxmum_items_number=maxmum_items_number)
        self.maxmum_items_number = maxmum_items_number

    def getInitState(self,maxmum_items_number):
        self.InitState = CSPState(instance_data=self.instance_data, maxmum_items_number=maxmum_items_number)
        return self.InitState

    def getStateSize(self):
        return (len(self.InitState.item_feature), len(self.InitState.item_feature[0]))

    def getActionSize(self):
        # return number of actions
        return self.InitState.action_size

    def getActionList(self, state):
        return range(state.action_size)

    def getNextState(self, state, action):
        # if player takes action on board, return next (board)
        # action must be a valid move
        #b=copy.deepcopy(board)
        newstate = state.getNextState(action)
        return newstate

    def getActionMask(self, state):
        return state.mask

    def getGameEnded(self, state):
        # return 0 if not ended, 1 ended
        #b = copy.deepcopy(board)
        if not state.getStateTerminal():
            return 0
        else:
            return self.getScore(state)

    def getCanonicalForm(self, state):
        return state.state

    def getSymmetries(self, board, pi):
        return 0

    def stringRepresentation(self, state):
        # numpy array (canonical board)
        return state.getStateLabel()


    def getScore(self, state):
        obj = state.getScore()
        return -obj
