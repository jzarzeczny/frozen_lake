import gymnasium as gym
import numpy as np


class Utils:
    def doSomething():
        print("hello")
    
    def getEnv(map, isTraining):
        print(map)
        render_mode = 'human' if not isTraining else None 
        return gym.make('FrozenLake-v1', desc=None, map_name=map, is_slippery=False,render_mode=render_mode)

    def epsilon_greedy_action_selection(epsilon,Q,state,env):
        random_number = np.random.random()
        if random_number > epsilon:
            action = np.argmax(Q[state,:])
        else:
            action = env.action_space.sample()
        return action
    
    def reduce_epsilon(epsilon,epoch, max_epsilon, min_epsilon):
        return min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate * epoch)

    def compute_next_q_value(old_q_value,reward,next_optimal_q_value):
        return old_q_value + ALPHA * (reward + GAMMA*next_optimal_q_value - old_q_value)
