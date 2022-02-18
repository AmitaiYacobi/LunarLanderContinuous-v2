import io
import gym
import glob
import base64
import matplotlib.pyplot as plt

from dqn import *
from ddqn import *
from sarsa import *
from targetdqn import *
from discritization import *
from gym.wrappers import Monitor
from IPython.display import HTML
from IPython import display as ipythondisplay

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[5]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay loop controls style="height: 400px;">
                                        <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                                        </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env



if __name__ == "__main__":

  # TODO: 1. add the option to get the algorithm as an argumnet 
  #       2. make conditions that will initialize the agent by the user choice (by the argument)
  #       3. add title to the graph figures with the algorithm name
  #       4. save the figure with the name of the algorithm
 
  env = gym.make("LunarLanderContinuous-v2")
  env = wrap_env(env)
  state_size = env.observation_space.shape[0]
  action_size = len(discrete_actions)

  agent = SARSAGENT()
  rewards = agent.solve(env, 1000)

  plt.xlabel("Episodes")
  plt.ylabel("Rewards")
  plt.plot([i + 1 for i in range(0, len(rewards), 2)], rewards[::2])
  plt.show()

        


