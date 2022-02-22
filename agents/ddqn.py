import torch
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network import *
from discritization import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DDQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.timestep = 0
        self.batch_size = 32
        self.buffer_size = 1000000
        self.lr = 0.0005
        self.gamma = .99
        self.tau = 0.01
        self.epsilon = 1.0
        self.epsilon_min = .01
        self.epsilon_decay = .996
        self.should_be_updated = 50

        #---------networks initialization--------#
        self.online_network = Network(state_size, action_size, seed).to(device)
        self.target_network = Network(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.lr)
        #----------memory initialization---------#
        self.memory = Memory(buffer_size=self.buffer_size, batch_size=self.batch_size, seed=seed)
    
    def step(self, state, action, reward, next_state, done):
        self.timestep += 1
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            sampled_experinces = self.memory.sample()
            self.learn(sampled_experinces)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        # Get max indicies of the output that was created by the online network on next states, 
        # then choose the actions from target output by those indices. This is actually how DDQN works.
        chosen_actions_for_next_states = self.online_network(next_states).detach().max(1)[1].unsqueeze(1)
        targets_for_next_states = self.target_network(next_states).detach().gather(1,chosen_actions_for_next_states)
        
        predicted = self.online_network(states).gather(1, actions)
        targets = rewards + (self.gamma * targets_for_next_states * (1 - dones))
        loss = F.mse_loss(predicted, targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.timestep % self.should_be_updated == 0:
            for target_param, local_param in zip(self.target_network.parameters(), self.online_network.parameters()):
                target_param.data.copy_(local_param.data)  

        # ----------------------- decay epsilon ------------------------ #       
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 

    def act(self, state):
        """
        Choose the action
        """
        rnd = random.random()
        if rnd <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.online_network.eval()
            with torch.no_grad():
                action_values = self.online_network(state)
            self.online_network.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action    
        
    def checkpoint(self, filename):
        torch.save(self.online_network.state_dict(), filename)

    def solve(self, env, num_of_episodes=1000):
        rewards = []
        is_finished = False
        for episode in range(num_of_episodes):
            state = env.reset()
            score = 0
            max_steps = 3000
            for _ in range(max_steps):
                action = self.act(state) # returns index of an action
                next_state, reward, done, _ = env.step(discrete_actions[action])
                env.render()
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    print(f"Episode: {episode}/{num_of_episodes}, score: {score}", end="\r")
                    break
            
            if score >= 200:
                print("\n")
                print("################################################################")
                print("Current score is 200! let's try 100 episodes to see if we are done!")
                print("################################################################")
                rewards_over_100 = []
                for e in range(100):
                    state = env.reset()
                    temp_score = 0
                    for _ in range(max_steps):
                        action = self.act(state) # returns index of an action
                        next_state, reward, done, _ = env.step(discrete_actions[action])
                        env.render()
                        state = next_state
                        temp_score += reward
                        if done:
                            print(f"Episode: {e}/100, score: {temp_score}", end="\r")
                            break
                    rewards_over_100.append(temp_score)

                result = np.mean(rewards_over_100)
                if result >= 200:
                    self.checkpoint('solved_200.pth')
                    print("\n")
                    print(f"Enviroment solved in {episode} episodes!")
                    is_finished = True
                else:
                    print(f"Enviroment not solved yet! Average score over 100: {result}")
                    
            rewards.append(score)
            if is_finished == True:
                break            
            result = np.mean(rewards[-100:])
            if episode % 100 == 0 and episode != 0: 
                print(f"Average score in episode {episode} is: {result}")
        
        return rewards




