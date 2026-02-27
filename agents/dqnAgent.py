from agents.abstractAgent import AbstractAgent
from pathlib import Path
from torch import nn

import numpy as np
import pandas as pd
import torch


class ReplayMemory:
    """
    Replay memory for off-policy agents: a simple buffer and a PER version
    """
    def __init__(self, capacity, batch_size, state_dim, is_prioritized=False, alpha=0.6, beta=0.4, beta_annealing=0.9999, device="cpu"):
        self.capacity = capacity
        self.batch_size = batch_size
        self.is_prioritized = is_prioritized
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.pos = 0
        self.device = device

        # Dimensions: State + Action + Reward + Next State + Done
        # Action (1) + Reward (1) + Done (1) = 3
        self.feature_size = (state_dim * 2) + 3
        self.state_dim = state_dim

        self.memory = torch.zeros((capacity, self.feature_size), dtype=torch.float32, device=device)
        if self.is_prioritized:
                    raise NotImplementedError()
        
    def add(self, state, action, reward, next_state, done):
        # Ensure inputs are flat lists/arrays
        transition = [*state, action, reward, *next_state, int(done)]
        self.memory[self.pos % self.capacity] = torch.tensor(transition, device=self.device)
        self.pos += 1

    def sample(self):
        # Wrap around if pos > capacity, otherwise sample up to current pos
        max_idx = min(self.pos, self.capacity)
        batch_inds = torch.randint(0, max_idx, size=(self.batch_size,), device=self.device)
        return self.memory[batch_inds]

    def __len__(self):
        return min(self.pos, self.capacity)

class DQN(nn.Module):
    def __init__(self, input_shape, l1_size, l2_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_shape, l1_size) 
        self.fc2 = nn.Linear(l1_size, l2_size) 
        self.out = nn.Linear(l2_size, output_size) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.out(x)
        
        return x

class DQNAgent(AbstractAgent):
    """
    DQN RL agent
    """
    def __init__(self, state_shape, action_shape,
                 batch_size=128,
                 learning_rate=1e-4,
                 discount_factor=0.99, 
                 epsilon=1.0,
                 epsilon_decay=0.999995,
                 epsilon_min=0.05,
                 net_arch=[64, 64],
                 target_update_freq=4000,
                 online_update_freq = 10,
                 memory_capacity=10000,
                 learn_after_steps=4000):
        super().__init__(state_shape, action_shape)
                
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.online_update_freq = online_update_freq
        self.learn_after_steps = learn_after_steps
        self.step_count = 0
        self.state_dim = state_shape[0]
        self.n_actions = int(action_shape[0] if isinstance(action_shape, (tuple, list)) else action_shape)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Agent running on device: {self.device}")

        self.online_dqn = DQN(self.state_dim, net_arch[0], net_arch[1], self.n_actions).to(self.device)
        self.target_dqn = DQN(self.state_dim, net_arch[0], net_arch[1], self.n_actions).to(self.device)
        self.target_dqn.load_state_dict(self.online_dqn.state_dict())

        self.loss = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.online_dqn.parameters(), lr=self.learning_rate)

        self.memory = ReplayMemory(memory_capacity, batch_size, self.state_dim, device=self.device)

    def get_action(self, state):
        if torch.rand(1).item() < self.epsilon:
            action = torch.randint(0, self.n_actions, size=(1,)).item()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.online_dqn(state_tensor)
                action = torch.argmax(q_values).item()

        self.step_count += 1
        return action
    
    def optimise(self, mini_batch):
        state_dim = self.state_dim
        
        states = mini_batch[:, :state_dim]
        actions = mini_batch[:, state_dim].long()
        rewards = mini_batch[:, state_dim + 1]
        next_states = mini_batch[:, state_dim + 2 : state_dim * 2 + 2]
        dones = mini_batch[:, -1]

        with torch.no_grad():
            # Double DQN target
            next_actions = self.online_dqn(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_dqn(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.discount_factor * next_q * (1 - dones)

        output = self.online_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss(output, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_dqn.parameters(), 10.0)
        self.optimizer.step()

    def update(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        # decay every step
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if self.step_count <= self.learn_after_steps or len(self.memory) < self.batch_size:
            return
        
        if self.step_count % self.online_update_freq == 0:
            mini_batch = self.memory.sample()
            self.optimise(mini_batch)

        if self.step_count % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.online_dqn.state_dict())

    def save_model(self, path, filename="dqn.pt"):
        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / filename
        torch.save(self.online_dqn.state_dict(), model_path)

    @classmethod
    def load_model(cls, path, filename="dqn.pt", reset_timesteps=False, load_memory=True):
        model_path = Path(path) / filename
        state_dict = torch.load(model_path)
        agent = cls(state_shape=(5,), action_shape=3)
        agent.online_dqn.load_state_dict(state_dict)
        agent.target_dqn.load_state_dict(state_dict)
        if reset_timesteps:
            agent.step_count = 0
        if not load_memory:
            agent.memory = ReplayMemory(agent.memory.capacity, agent.memory.batch_size, agent.memory.state_dim, device=agent.device)
        return agent