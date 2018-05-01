import random
from collections import deque

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make('SpaceInvaders-v0')

env.render()


def wrap_state(state):
    """It wraps state in a tensor."""
    return torch.tensor(state).view(3, 210, 160).unsqueeze(0).float()


class DQN(nn.Module):
    """A NN from state to actions."""

    def __init__(self, num_actions):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        no_lstm_layers = 6
        self.lstm_hidden = (torch.rand(no_lstm_layers, 1, 256),
                            torch.rand(no_lstm_layers, 1, 256))
        self.lstm = nn.LSTM(22528, 256, no_lstm_layers)

        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        y, self.lstm_hidden = self.lstm(x.view(batch_size, 1, -1),
                                        self.lstm_hidden)
        self.lstm_hidden = (self.lstm_hidden[0].detach(),
                            self.lstm_hidden[1].detach())

        return self.fc2(y.view(batch_size, 256))


class DQNAgent():
    """It uses DQN and experience replay."""

    def __init__(self,
                 num_actions,
                 gamma, alpha,
                 buffer_size, batch_size,
                 epsilon):
        self.dqn = DQN(num_actions)
        self.num_actions = num_actions

        self.criterion = nn.MSELoss()
        self.gamma = torch.tensor(gamma)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=alpha)

        self.batch_size = batch_size

        t = wrap_state(env.reset())
        self.buffer = deque(buffer_size * [(t, 0., t)], buffer_size)

        self.epsilon = epsilon

    def update(self):
        batch = random.sample(self.buffer, self.batch_size)
        state1, reward, state2 = ([], [], [])
        for state1_, reward_, state2_ in batch:
            state1.append(state1_)
            reward.append(reward_)
            state2.append(state2_)
        state1 = torch.cat(state1)
        reward = torch.tensor(reward).view(-1, 1)
        state2 = torch.cat(state2)

        target = reward + self.gamma * self.dqn(state2).max(1)[0].view(-1, 1)
        target = torch.cat([target for _ in range(self.num_actions)], 1)
        Q = self.dqn(state1)
        loss = self.criterion(target, Q.detach())
        loss.backward()
        self.optimizer.step()

    def epsilon_greedy(self, state):
        action = 0
        if torch.rand(1)[0] > epsilon:
            action = env.action_space.sample()
        else:
            action = self.dqn(state).max(1)[1].item()
        return action


if __name__ == '__main__':
    alpha, gamma, epsilon = (0.65, 0.65, 0.875)

    agent = DQNAgent(env.action_space.n,
                     gamma, alpha,
                     30, 5,
                     epsilon)

    for episode in range(1, 201):
        done = False
        G, reward = 0, 0

        state1 = wrap_state(env.reset())
        while done is not True:
            action = agent.epsilon_greedy(state1)
            state2, reward, done, info = env.step(action)

            state2 = wrap_state(state2)

            agent.buffer.append((state1, reward, state2))
            agent.update()

            state1 = state2

            G += reward
            env.render()

        if episode % 50 == 0:
            print("Episode {}: Total reward = {}.".format(episode, G))
