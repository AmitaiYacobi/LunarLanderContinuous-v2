import argparse

from agents.dqn import *
from agents.ddqn import *
from agents.sarsa import *
from agents.targetdqn import *
from agents.tdqn_vdiff import *

algorithms_dictionary = {
  "sarsa": SARSAgent,
  "dqn": DQNAgent,
  "targetdqn": TargetDQNAgent,
  "ddqn": DDQNAgent,
  "tdqn_vdiff" : TDQNVD_Agent
}


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "agent",
    type=str,
    help="List of optional agents you can choose",
    choices = [i for i in algorithms_dictionary],
  )
  args = parser.parse_args()

  env = gym.make("LunarLanderContinuous-v2")
  # env = wrap_env(env)
  state_size = env.observation_space.shape[0]
  action_size = len(discrete_actions)

  print(args.agent)
  if args.agent == "sarsa":
    agent = algorithms_dictionary[args.agent]()
  elif args.agent == "tdqn_nv":
    agent = algorithms_dictionary[args.agent](state_size - 2, action_size, 0)
  else:
    agent = algorithms_dictionary[args.agent](state_size, action_size, 0)

  rewards = agent.solve(env, 1000)

  plt.title(args.agent.upper())
  plt.xlabel("Episodes")
  plt.ylabel("Rewards")
  plt.plot([i + 1 for i in range(0, len(rewards), 2)], rewards[::2])
  plt.show()
  plt.savefig(args.agent.upper())

        


