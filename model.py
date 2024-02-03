import torch
import torch.nn as nn
import torch.nn.functional as F


class HexapawnNet(nn.Module):
    def __init__(self, state) -> None:
        self.state = state
        self.inp_dim = state.shape[0]
        self.action_size = 28

        super(HexapawnNet, self).__init__()
        self.fc = nn.Linear(in_features=128, out_features=128)

    def forward(self, state):
        state = state.view(-1, self.inp_dim)

        state = nn.Linear(in_features=self.inp_dim, out_features=128)(state)
        state = F.relu(self.fc(state))
        state = F.relu(self.fc(state))
        state = F.relu(self.fc(state))
        state = F.relu(self.fc(state))

        pi = nn.Linear(in_features=128, out_features=self.action_size)(state)
        v = nn.Linear(128, 1)(state)

        return F.softmax(pi, dim=0), torch.tanh(v)
