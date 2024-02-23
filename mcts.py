import numpy as np
import torch

from game import is_game_over, get_move_idx, get_valid_moves, make_move, draw_board
from utils import get_input_from_state, MOVE_INDEX
from model import HexapawnNet


class Node:
    def __init__(self, state) -> None:
        self.state = np.array(state).reshape(3, 3)
        self.player = None
        self.num_visits = 0
        self.q_value = 0
        self.policy = [0] * 28
        self.parent = None
        self.children = []

    def __repr__(self):
        return f"Node({self.state.flatten()}, visits = {float(self.num_visits)})"

    def expand(self, player):
        moves = get_valid_moves(self.state, player)
        for m in moves:
            child_state = make_move(self.state.copy(), m[0], m[1], player)
            child = Node(child_state)
            child.parent = self
            self.children.append(child)

    def get_mcts_policy(self, player):
        policy = torch.zeros(28)
        total_visits = sum([child.num_visits for child in self.children])
        for child in self.children:
            idx = get_move_idx(self, child, player)
            policy[idx] = child.num_visits / total_visits

        # if sum(policy) != 1:
        #     draw_board(self.state)
        #     print(self.policy)
        #     print(self.player, self.state)
        #     print(is_game_over(self.state, self.player * -1))

        return policy


def mask_illegal_moves(state, policy):
    policy.detach().numpy().copy()
    valid_moves = get_valid_moves(state.state, state.player)
    # Mask illegal move
    for idx in MOVE_INDEX:
        if MOVE_INDEX[idx] not in valid_moves:
            policy[idx] = 0

    return policy / sum(policy)


def mcts(state, player, hnet):
    state.player = player

    game_over, reward = is_game_over(state.state, player)
    if game_over:
        # state.player = - player
        state.q_value = (state.num_visits * state.q_value + (- reward)) / (state.num_visits + 1)
        state.num_visits += 1
        return - reward

    if state.num_visits == 0:
        state.num_visits += 1
        policy, value = hnet.predict(get_input_from_state(state.state, player))
        state.policy = mask_illegal_moves(
            state, torch.exp(policy)    # because model retuns log softmax
        )
        # state.policy = torch.exp(policy)
        return - value

    max_u, best_child = -float("inf"), None

    if not state.children:
        state.expand(player)

    for c in state.children:
        idx = get_move_idx(state, c, player)
        u_value = (
            c.q_value
            + 1 * state.policy[idx] * np.sqrt(state.num_visits) / (1 + c.num_visits)
        )

        if u_value > max_u:
            max_u = u_value
            best_child = c

    v = mcts(best_child, - player, hnet)

    state.q_value = (state.num_visits * state.q_value + v) / (state.num_visits + 1)
    state.num_visits += 1
    return -v


if __name__ == "__main__":
    # from game import draw_board

    state = np.array([
        [-1, 0, -1],
        [0,  -1,  1],
        [1,  0,  0]
    ])

    state = Node(state)
    # print(state, state.num_visits, state.player)
    draw_board(state.state)
    print("----------------------")

    hnet = HexapawnNet()
    for _ in range(10):
        mcts(state, player=-1, hnet=hnet)

    print(state.get_mcts_policy(state.player))
    print("----------------------")

    for c in state.children:
        draw_board(c.state)
        print(c.q_value, c.num_visits)
        # print(c.children, c.player)

    # print(state.get_mcts_policy(-1))
