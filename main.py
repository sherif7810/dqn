import gym

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make('SpaceInvaders-v0')

env.render()


class DQN(nn.Module):
    """A NN from state to actions."""
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.lstm_hidden = (Variable(torch.rand(6, 1, 256)),
                            Variable(torch.rand(6, 1, 256)))
        self.lstm = nn.LSTM(22528, 256, 6)

        self.fc2 = nn.Linear(256, self.num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        y, self.lstm_hidden = self.lstm(x.view(1, 1, -1), self.lstm_hidden)
        self.lstm_hidden = (Variable(self.lstm_hidden[0].data),
                            Variable(self.lstm_hidden[1].data))

        return self.fc2(y.view(1, 256))


alpha, gamma, epsilon = (0.65, 0.65, 0.875)
model = DQN(env.action_space.n)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=alpha)


def epsilon_greedy(state):
    action = 0
    Q = model(state)
    if torch.rand(1)[0] > epsilon:
        action = env.action_space.sample()
    else:
        action = Q.data.max(1)[1]
    return (action, Q)


for episode in range(1, 201):
    done = False
    G, reward = 0, 0

    state1 = Variable(torch.Tensor(env.reset()).view(3, 210, 160)).unsqueeze(0)
    while done is not True:
        action, Q1 = epsilon_greedy(state1)
        state2, reward, done, info = env.step(action)

        state2 = Variable(torch.Tensor(state2).view(3, 210, 160)).unsqueeze(0)
        Q_, index = model(state2).data.max(1)
        Q_list = [torch.Tensor([0.0]) for i in range(env.action_space.n)]
        Q_list[int(index)] = reward + gamma * Q_
        Q_list = torch.cat(Q_list)
        Q1 = Variable(Q1.data)
        loss = criterion(Variable(Q_list, requires_grad=True),
                         Q1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state1 = state2

        G += reward

        env.render()

    if episode % 50 == 0:
        print("Episode {}: Total reward = {}.".format(episode, G))
