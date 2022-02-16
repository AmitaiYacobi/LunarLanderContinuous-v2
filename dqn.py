import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

""" This implementation of DQN algorithm uses replay memory and epsilon greedy selection method """


class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """
        Build a fully connected neural network
        """
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 150)
        self.fc2 = nn.Linear(150, 120)
        self.fc3 = nn.Linear(120, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Memory:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = seed
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert to torch tensors
        states = torch.from_numpy(
            np.vstack([experience.state for experience in experiences if experience is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([experience.action for experience in experiences if experience is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([experience.reward for experience in experiences if experience is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([experience.next_state for experience in experiences if experience is not None])).float().to(
            device)
        # Convert done from boolean to int
        dones = torch.from_numpy(
            np.vstack([experience.done for experience in experiences if experience is not None]).astype(
                np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.timestep = 0
        self.batch_size = 64
        self.buffer_size = 1000000
        self.lr = 0.0005
        self.gamma = .99
        self.tau = 0.01
        self.epsilon = 1.0
        self.epsilon_min = .01
        self.epsilon_decay = .996
        self.should_be_updated = 50

        # ---------networks initialization--------#
        self.network = DQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        # ----------memory initialization---------#
        self.memory = Memory(buffer_size=self.buffer_size, batch_size=self.batch_size, seed=seed)

    def step(self, state, action, reward, next_state, done):
        """
        Add current sample to the memory (replay buffer).
        If there are enough samples - use them to train the network
        """
        self.timestep += 1
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            sampled_experinces = self.memory.sample()
            self.learn(sampled_experinces)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        targets_next = self.network(next_states).max(1)[0].detach().unsqueeze(1)
        targets = rewards + (self.gamma * targets_next * (1 - dones))
        predicted = self.network(states).gather(1, actions)
        loss = F.mse_loss(predicted, targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action

    def checkpoint(self, filename):
        torch.save(self.network.state_dict(), filename)
