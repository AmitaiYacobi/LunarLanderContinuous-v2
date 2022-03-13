import gym
import random
import collections
import numpy as np
import matplotlib.pyplot as plt

from discritization import *


class SARSAgent:
    def __init__(self):
        self.alpha = 0.3
        self.gamma = 0.95
        self.epsilon = 50
        self.epsilon_decay = 0.8
        self.Q = collections.defaultdict(float)


    def state_discritization(self, state):
        new_state = (
                     min(2, max(-2, int((state[0]) / 0.05))), \
                     min(2, max(-1, int((state[1]) / 0.1))), \
                     min(1, max(-1, int((state[2]) / 0.1))), \
                     min(1, max(-1, int((state[3]) / 0.1))), \
                     min(1, max(-1, int((state[4]) / 0.1))), \
                     min(1, max(-1, int((state[5]) / 0.1))), \
                     int(state[6]), \
                     int(state[7])
                    )

        return new_state

    def state_action_key(self, state, action):
        return str(state) + " " + str(action)

    def choose_action(self, state, Q, epsilon):
        rand = np.random.randint(0, 100)
        if rand >= epsilon:
            Qv = np.array([Q[self.state_action_key(state, action)] for action in range(0, 15)])
            action =  np.argmax(Qv)
        else:
            action =  np.random.randint(0, 15)
        return action

    def solve(self, env, num_episodes=10000, batch=100, render=False):
        gamma = self.gamma
        env.seed(42)
        rewards= []
        episode_reward = []

        for episode in range(num_episodes):
            total_reward = 0
            max_steps = 1000
            alpha = self.alpha
            state = env.reset()
            discrete_state = self.state_discritization(state)
            action = self.choose_action(discrete_state, self.Q, self.epsilon)

            for _ in range(max_steps):
                state_action = self.state_action_key(discrete_state, action)
                next_state, reward, done, info = env.step(discrete_actions[action])

                discrete_next_state = self.state_discritization(next_state)
                next_action = self.choose_action(discrete_next_state, self.Q, self.epsilon)
                next_state_action = self.state_action_key(discrete_next_state, next_action)

                if not done:
                    self.Q[state_action] += alpha * (reward + gamma * (self.Q[next_state_action] - self.Q[state_action]))
                else:
                    self.Q[state_action] += alpha * (reward - self.Q[state_action])

                discrete_state = discrete_next_state
                action = next_action
                total_reward += reward

                if done:
                    episode_reward.append(total_reward)
                    break
            
            if episode % 500 == 0 and episode > 0:
                self.epsilon *= self.epsilon_decay 
             

            # if episode > 200:
            #     self.epsilon = 10
            # if episode > 2000:
            #     self.epsilon = 5
            # if episode > 5000:
            #     self.epsilon = 1
            # if episode > 7500:
            #     self.epsilon = 0

            if episode % batch == 0:
                avg_reward = np.mean(np.array(episode_reward))
                print("Episode: ", episode, " avg reward: ", avg_reward)
                episode_reward = []
                rewards.append(avg_reward)

        return rewards


