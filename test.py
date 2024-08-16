import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

from utils import Utils

class Test():

    def __init__(self, map, mode):
        self.map = map
        self.mode = mode
    
    def run(self):
        env = Utils.getEnv(self.map, False)
        q_table = np.load('Q.npy')
        state,info = env.reset()
        reward=0
        round=0
        for steps in range(100):
            env.render()
            action = np.argmax(q_table[state,:])
            state,reward,truncated,terminated,info = env.step(action)
            
            time.sleep(0.5)    
            if truncated or terminated:
                round += 1
                print(f"reward at round: {round} is = {reward}")
                state,info = env.reset()
        env.close()