import gym
import random
import collections
import numpy as np
import matplotlib.pyplot as plt

from discritization import *


class SARSAgent:
    def state_discritization(self, state):
        new_state = (min(2, max(-2, int((state[0]) / 0.05))), \
                     min(2, max(-1, int((state[1]) / 0.1))), \
                     min(1, max(-1, int((state[2]) / 0.1))), \
                     min(1, max(-1, int((state[3]) / 0.1))), \
                     min(1, max(-1, int((state[4]) / 0.1))), \
                     min(1, max(-1, int((state[5]) / 0.1))), \
                     int(state[6]), \
                     int(state[7])
                    )

        return new_state

    def alpha(self, it):
        return 0.3

    def state_action_key(self, state, action):
        return str(state) + " " + str(action)

    def policy_explorer(self, state, Q, episode):
        rand = np.random.randint(0, 100)
        epsilon = 50
        if episode > 200:
            epsilon = 10
        if episode > 2000:
            epsilon = 5
        if episode > 5000:
            epsilon = 1
        if episode > 7500:
            epsilon = 0
        if rand >= epsilon:
            Qv = np.array([Q[self.state_action_key(state, action)] for action in range(0, 15)])
            return np.argmax(Qv)
        else:
            return np.random.randint(0, 4)

    def solve(self, env, num_episodes=1000, batch=100, render=False):
        gamma = 0.95
        env.seed(42)
        Q = collections.defaultdict(float)
        rewards= []
        episode_reward = []

        for episode in range(num_episodes):
            total_reward = 0
            max_steps = 1000
            alpha = self.alpha(episode)
            state = env.reset()
            discrete_state = self.state_discritization(state)
            action = self.policy_explorer(discrete_state, Q, episode)
            for _ in range(max_steps):
                state_action = self.state_action_key(discrete_state, action)
                next_state, reward, done, info = env.step(discrete_actions[action])

                discrete_next_state = self.state_discritization(next_state)
                next_action_according_to_policy = self.policy_explorer(discrete_next_state, Q, episode)
                next_state_action = self.state_action_key(discrete_next_state, next_action_according_to_policy)

                # sarsa update rule
                if not done:
                    Q[state_action] += alpha * (reward + gamma * (Q[next_state_action] - Q[state_action]))
                else:
                    Q[state_action] += alpha * (reward - Q[state_action])

                discrete_state = discrete_next_state
                action = next_action_according_to_policy
                total_reward += reward

                if render and episode % batch == 0:
                    still_open = env.render()
                    if still_open == False: break

                if done:
                    episode_reward.append(total_reward)
                    break

            if episode % batch == 0:
                avg_reward = np.mean(np.array(episode_reward))
                print("Episode: ", episode, " avg reward: ", avg_reward)
                episode_reward = []
                rewards.append(avg_reward)

        return rewards


# if __name__ == '__main__':
#     env = gym.make("LunarLanderContinuous-v2")
#     agent = SARSA() 
#     num_episodes = 10000
#     reward_seq = agent.solve(env, num_episodes=num_episodes, batch=100)

#     y = np.array(reward_seq)
#     x = np.linspace(0, num_episodes, y.shape[0])

#     plt.title("SARSA, alpha = 0.3, gamma = 0.95")
#     plt.xlabel("Episodes")
#     plt.ylabel("Rewards")
#     plt.plot(x, y)
#     plt.savefig("figures/sarsa.png")
