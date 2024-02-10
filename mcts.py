import numpy as np

from game import is_game_over, get_move_idx, get_valid_moves, make_move
from utils import get_input_from_state


class Node:
    def __init__(self, state) -> None:
        self.state = np.array(state).reshape(3, 3)
        self.num_visits = 0
        self.num_wins = 0
        self.q_value = 0
        self.parent = None
        self.children = []

    def __repr__(self):
        return f"Node({self.state.flatten()}, value = {self.value})"

    def __eq__(self, other):
        return (self.state == other.state).all()

    def expansion(self, player):
        moves = get_valid_moves(self.state)
        for m in moves:
            child = Node(make_move(self.state, m[0], m[1], player))
            child.parent = self
        self.children.append(child)


def mcts(state, player, hnet):
    game_over, reward = is_game_over(state, player)
    if game_over:
        return - reward

    if state.num_visits == 0:
        state.num_visits += 1
        policy, value = hnet.predict(get_input_from_state(state, player))
        return - value

    max_u, best_child = -float("inf"), -1
    state.expansion(player)

    for c in state.children:
        idx = get_move_idx(state, c, policy, player)
        u_value = c.q_value + 0.1 * policy[idx] * np.sqrt(state.num_visits) / (1 + c.num_visits)

        if u_value > max_u:
            max_u = u_value
            best_child = c

    v = mcts(best_child, - player, hnet)

    state.num_visits += 1
    state.q_value = (state.num_visits * state.q_value + v) / (state.num_visits + 1)
