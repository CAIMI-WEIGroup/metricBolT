
from datetime import datetime
import copy
import os
import torch

class Option(object):
      
    def __init__(self, my_dict):

        self.dict = my_dict

        for key in my_dict:
            setattr(self, key, my_dict[key])

    def copy(self):
        return Option(copy.deepcopy(self.dict))
