import random
from collections import deque

import gym

import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make('SpaceInvaders-v0')

env.render()


def wrap_state(state) -> torch.Tensor:
    """It wraps state in a `Tensor` then normalizes it to [0, 1]."""
    hw = state.shape[:2]
    return torch.FloatTensor(state).view(3, hw[0], hw[1]).unsqueeze(0).float() / 255.0


class DQN(torch.jit.ScriptModule):
    """A NN from state to actions."""

    def __init__(self,
                 env,
                 gamma: float, alpha: float,
                 buffer_size: int, batch_size: int,
                 epsilon: float):
        super(DQN, self).__init__()

        self.batch_size = batch_size

        self.gamma = torch.tensor(gamma)
        self.epsilon = torch.jit.Attribute(epsilon, float)

        t = wrap_state(env.reset())
        self.buffer = deque(buffer_size * [(t, 0, 0., t)], buffer_size)

        action_space = torch.tensor(range(env.action_space.n)).float()
        self.action_distribution = torch.distributions.Categorical(action_space)

        conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        relu = nn.ReLU()
        self.conv_layers = nn.Sequential(
            conv1, relu,
            conv2, relu,
            conv3, relu
        )

        self.no_lstm_layers = torch.jit.Attribute(6, int)
        self.lstm = nn.LSTM(22528, 256, self.no_lstm_layers)

        self.fc = nn.Sequential(
            nn.Linear(256, env.action_space.n),
            nn.Sigmoid()
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), alpha)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_hidden = (torch.zeros(self.no_lstm_layers, 1, 256),
                       torch.zeros(self.no_lstm_layers, 1, 256))

        x = self.conv_layers(x)

        batch_size = x.size(0)
        x, lstm_hidden = self.lstm(x.view(batch_size, 1, -1),
                                   lstm_hidden)

        x = x.view(batch_size, 256)
        x = self.fc(x)
        return x

    def update(self):
        """Update agent."""

        batch = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state = ([], [], [], [])
        for state_, action_, reward_, next_state_ in batch:
            state.append(state_)
            action.append([action_])  # To give it the same dimension of dqn.
            reward.append(reward_)
            next_state.append(next_state_)
        state = torch.cat(state)
        action = torch.tensor(action)
        reward = torch.tensor(reward).view(-1, 1)
        next_state = torch.cat(next_state)

        target = reward + self.gamma * self(next_state).max(1)[0].view(-1, 1)
        Q = self(state).gather(1, action)
        loss = self.criterion(Q, target.detach())
        loss.backward()
        self.optimizer.step()

    def epsilon_greedy(self, state: torch.Tensor) -> int:
        action = None
        if torch.rand(1)[0] > self.epsilon:
            action = self.action_distribution.sample().item()
        else:
            action = torch.argmax(self(state), 1).item()
        return action


if __name__ == '__main__':
    alpha, gamma, epsilon = (0.65, 0.65, 0.875)

    agent = DQN(env,
                gamma, alpha,
                30, 5,
                epsilon)

    for episode in range(1, 101):
        done = False
        G, reward = 0, 0

        state = wrap_state(env.reset())
        while done is not True:
            action = agent.epsilon_greedy(state)
            next_state, reward, done, info = env.step(action)
            next_state = wrap_state(next_state)

            agent.buffer.append((state, action, reward, next_state))
            agent.update()

            state = next_state

            G += reward
            env.render()

        if episode % 10 == 0:
            print("Episode {}: Total reward = {}.".format(episode, G))
