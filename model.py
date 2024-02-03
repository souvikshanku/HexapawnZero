import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class HexapawnNet(nn.Module):
    def __init__(self) -> None:
        self.inp_dim = 21
        self.action_size = 28
        self.epochs = 10

        super(HexapawnNet, self).__init__()
        self.fc1 = nn.Linear(in_features=self.inp_dim, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=128)

        self.fc5 = nn.Linear(in_features=128, out_features=self.action_size)
        self.fc6 = nn.Linear(128, 1)

    def forward(self, state):
        state = state.view(-1, self.inp_dim)

        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        state = F.relu(self.fc3(state))
        state = F.relu(self.fc4(state))

        pi = self.fc5(state)
        v = self.fc6(state)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def train(self, examples):
        """
        examples: list[(board, pi, v)]
        """
        optimizer = optim.Adam(self.parameters())

        for _ in range(self.epochs):
            states, target_pis, target_vs = list(zip(*examples))
            states = torch.FloatTensor(np.array(states))
            target_pis = torch.FloatTensor(np.array(target_pis))
            target_vs = torch.FloatTensor(np.array(target_vs))

            pi, v = self.forward(states)

            l_pi = self.loss_pi(target_pis, pi)
            l_v = self.loss_v(target_vs, v)
            total_loss = l_pi + l_v

            print(total_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    def loss_pi(self, targets, outputs):
        return - torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


if __name__ == "__main__":
    examples = [
        (np.random.randn(21), np.random.randint(2, size=28), np.random.uniform()) for _ in range(20)
    ]

    hnet = HexapawnNet()
    hnet.train(examples)
