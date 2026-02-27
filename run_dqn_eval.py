from absl import flags
from agents.dqnAgent import DQNAgent
from env.env_full import FullEnv
from runner.runner import Runner

import sys
import torch


FLAGS = flags.FLAGS
FLAGS(sys.argv)

env = FullEnv(isVisualised=True)
agent = DQNAgent.load_model("models/260227_2250_DQNAgent/dqn.pt")
runner = Runner(
    agent=agent,
    env=env,
    is_training=False)

runner.run(50)