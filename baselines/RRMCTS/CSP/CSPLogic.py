import copy

import numpy

import CSP.data.instance.functional_library as func_lib
import numpy as np

class CSPState:
    def __init__(self, instance_data, maxmum_items_number):
        self.instance_data = instance_data
        self.action_size = maxmum_items_number
        self.item_width_sequence = list()
        self.state = copy.deepcopy(np.array(self.instance_data)[:, np.newaxis])
        self.mask = np.squeeze(self.state > 0)

    def getNextState(self, action):
        newstate = copy.deepcopy(self)
        newstate.item_width_sequence.append(self.instance_data[action])
        newstate.mask[action] = 0
        return newstate

    def getActionMask(self):
        return self.mask

    def getScore(self):
        obj, sum_length, ele_idx = 0, 0, 0
        while ele_idx < len(self.item_width_sequence)-1:
            if sum_length+self.item_width_sequence[ele_idx] >1:
                obj += 1
                sum_length = 0
            else:
                sum_length += self.item_width_sequence[ele_idx]
                ele_idx += 1

        return obj

    def getStateTerminal(self):
        if 0 == sum(self.mask):
            return True
        else:
            return False

    def getStateLabel(self):
        return '0{}'.format(np.array(self.item_width_sequence).tostring())