from absl import flags
from agents.dqnAgent import DQNAgent
from env.env_full import FullEnv
from runner.runner import Runner

import sys


FLAGS = flags.FLAGS
FLAGS(sys.argv)

env = FullEnv(isVisualised=False)
agent = DQNAgent(
    state_shape=env.observation_space.shape,
    action_shape=env.action_space.n,
    learn_after_steps=1000)
runner = Runner(
    agent=agent,
    env=env,
    save_model_each_episode_num=100,
    is_training=True)

runner.run(2000)