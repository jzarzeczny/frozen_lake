import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from train import Train
from test import Test


class Main():
    
   
    def __init__(self, map=4 , mode="test"):
        self.map = map
        self.mode = mode
        self.selectedMap = ""
    
    def run(self):
        self.setCorrectMap()
        self.runScenario()

    def runScenario(self):
        if(self.mode == "train"):
            train = Train(self.selectedMap, self.mode)
            train.run()


        elif(self.mode == "test"):
            test = Test(self.selectedMap, self.mode)
            test.run()
        
        else:
            raise "Incorrect second argument, provide 'train' or 'test'."

    def setCorrectMap(self):
        if self.map == 4:
            self.selectedMap = "4x4"
        elif self.map == 8:
            self.selectedMap = '8x8'
        else:
            raise "Incorrect first argument, provide 4 for 4x4 map or 8 for 8x8 map."


providedMap = int(sys.argv[1])
providedMode = sys.argv[2]
main =  Main(providedMap, providedMode)

main.run()
 
        