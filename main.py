import io
import gym
import glob
import base64
import numpy as np
import matplotlib.pyplot as plt

from ddqn import *
from gym.wrappers import Monitor
from IPython.display import HTML
from IPython import display as ipythondisplay




def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[5]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

env = gym.make("LunarLanderContinuous-v2")
env = wrap_env(env)

#-----discritization-----#
main_engine_values = [0, 0.5, 1]
sec_engine_values = [-1, -0.75, 0, 0.75, 1]
discrete_actions = [(x, y) for x in main_engine_values for y in sec_engine_values]
#------------------------#

state_size = env.observation_space.shape[0]
action_size = len(discrete_actions)

agent = Agent(state_size, action_size, 0)

num_of_episodes = 1000
rewards = []
for episode in range(num_of_episodes):
    state = env.reset()
    score = 0
    max_steps = 3000
    for _ in range(max_steps):
        action = agent.act(state) # returns index of an action
        next_state, reward, done, _ = env.step(discrete_actions[action])
        env.render()
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            print(f"Episode: {episode}/{num_of_episodes}, score: {score}", end="\r")
            break
    rewards.append(score)
    is_solved = np.mean(rewards[-100:])
    if is_solved >= 200:
        agent.checkpoint('solved_200.pth')
        print("\n")
        print(f"Enviroment solved in {episode} episodes!")
        break
    if episode % 100 == 0 and episode != 0: 
        print(f"Average score in episode {episode} is: {is_solved}")

plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.plot([i + 1 for i in range(0, len(rewards), 2)], rewards[::2])
plt.savefig("ddqn")
plt.show()

        

