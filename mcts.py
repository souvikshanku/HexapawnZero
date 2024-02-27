import numpy as np
import torch

from game import is_game_over, get_valid_moves, make_move
from utils import MOVE_INDEX, get_input_from_state, get_move_index
from model import HexapawnNet


def mask_illegal_moves(state: np.ndarray, policy: torch.Tensor, player: int):
    policy = policy.detach().numpy().copy()
    valid_moves = get_valid_moves(state, player)
    # Mask illegal move
    for idx in MOVE_INDEX:
        if MOVE_INDEX[idx] not in valid_moves:
            policy[idx] = 0

    return policy / sum(policy)


class MCTS:
    def __init__(self, hnet: HexapawnNet):
        self.hnet = hnet
        self.Qsa = {}  # stores Q values for s, a
        self.Nsa = {}  # stores #times edge s, a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

    def search(self, state, player):
        s = str(get_input_from_state(state, player))
        # s = draw_board(state) + str(player)

        if is_game_over(state, player)[0]:
            reward = is_game_over(state, player)[1]
            return - reward

        if s not in self.Ps:
            policy, value = self.hnet.predict(get_input_from_state(state, player))
            self.Ps[s] = mask_illegal_moves(state, policy, player)
            self.Ns[s] = 0
            return - value

        max_u, best_move = -float("inf"), None

        for move in get_valid_moves(state, player):
            a = get_move_index(move)
            s_a = str((s, a))

            if s_a in self.Qsa:
                u = (
                    self.Qsa[(s, a)] 
                    + 1 * self.Ps[s][a] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                )
            else:
                u = 1 * self.Ps[s][a] * np.sqrt(self.Ns[s] + 1e-8)  # Q = 0 ?

            if u > max_u:
                max_u = u
                best_move = move

        next_state = make_move(state, best_move[0], best_move[1], player)

        v = self.search(next_state, - player)

        s_a = str((s, best_move[0], best_move[1]))

        if s_a in self.Qsa:
            self.Qsa[s_a] = (self.Nsa[s_a] * self.Qsa[s_a] + v) / (self.Nsa[s_a] + 1)
            self.Nsa[s_a] += 1

        else:
            self.Qsa[s_a] = v
            self.Nsa[s_a] = 1

        self.Ns[s] += 1

        return - v


if __name__ == "__main__":
    hnet = HexapawnNet()
    s = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])

    mcts = MCTS(hnet)

    for _ in range(10):
        mcts.search(s, 1)

    print(mcts.Qsa)
    print(mcts.Ps)
    print(mcts.Nsa)
    print(mcts.Ns)
