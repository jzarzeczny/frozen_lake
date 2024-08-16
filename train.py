import numpy as np
import matplotlib.pyplot as plt

from utils import Utils

class Train():

    def __init__(self, map, mode):
        self.map = map
        self.mode = mode
        self.Q = []
        self.Alpha = 0.8
        self.Gamma = 0.9
        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.00001
        self.epoch = 1000
        self.rewards = []
        self.episodes = 100000

    
    def run(self):
        self.env = Utils.getEnv(self.map, True)
        self.Q = np.zeros([self.env.observation_space.n,self.env.action_space.n])
        self.train()
        self.saveTrainData()
        self.displayResults()

    def train(self):
        cum_reward=0
        for episode in range(self.episodes):
            state,info = self.env.reset()

            truncated = False
            terminated = False
            while not (truncated or terminated):
                action = self.epsilon_greedy_action_selection(self.epsilon, self.Q, state)
                
                next_state,reward,truncated,terminated,info = self.env.step(action)
                # calculate the Q- value
                self.Q[state,action] = self.Q[state,action] + self.Alpha*(reward + self.Gamma*np.max(self.Q[next_state,:])- self.Q[state,action])                                     
                state = next_state 
                cum_reward += reward 
                if episode%500 == 0 :
                    print(f'total cumulative rewards at episode {episode} is {cum_reward}')
                    break
            self.epsilon = self.reduce_epsilon(self.epsilon,episode)
            self.rewards.append(cum_reward)

    def epsilon_greedy_action_selection(self,epsilon,Q,state):
        random_number = np.random.random()
        if random_number > epsilon:
            action = np.argmax(Q[state,:])
        else:
            action = self.env.action_space.sample()
        return action
    
    def reduce_epsilon(self, epsilon,epoch):
        return self.min_epsilon + (self.max_epsilon-self.min_epsilon)*np.exp(-self.decay_rate * epoch)

    def compute_next_q_value(self, old_q_value,reward,next_optimal_q_value):
        return old_q_value + self.ALPHA * (reward + self.GAMMA*next_optimal_q_value - old_q_value)


    def saveTrainData(self):
        self.env.close()  
        np.save('Q.npy', self.Q)   

    def displayResults(self):
        plt.plot(range(self.episodes),self.rewards)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.title('reward vs episode')
        plt.show()
    



