import numpy as np

from game import is_game_over, get_move_idx, get_valid_moves, make_move
from utils import get_input_from_state
from model import HexapawnNet


class Node:
    def __init__(self, state) -> None:
        self.state = np.array(state).reshape(3, 3)
        self.num_visits = 0
        self.q_value = 0
        self.policy = [0] * 28
        self.parent = None
        self.children = []

    def __repr__(self):
        return f"Node({self.state.flatten()}, value = {float(self.q_value)})"

    def expansion(self, player):
        moves = get_valid_moves(self.state, player)
        for m in moves:
            child_state = make_move(self.state.copy(), m[0], m[1], player)
            child = Node(child_state)
            child.parent = self
            self.children.append(child)


def mcts(state, player, hnet):
    game_over, reward = is_game_over(state.state, player)
    if game_over:
        return - reward

    if state.num_visits == 0:
        state.num_visits += 1
        policy, value = hnet.predict(get_input_from_state(state.state, player))
        state.policy = policy
        return - value

    max_u, best_child = -float("inf"), None

    if not state.children:
        state.expansion(player)

    for c in state.children:
        idx = get_move_idx(state, c, player)
        u_value = (
            c.q_value
            + 0.1 * state.policy[idx] * np.sqrt(state.num_visits) / (1 + c.num_visits)
        )

        if u_value > max_u:
            max_u = u_value
            best_child = c

    v = mcts(best_child, - player, hnet)

    state.num_visits += 1
    state.q_value = (state.num_visits * state.q_value + v) / (state.num_visits + 1)
    return -v


if __name__ == "__main__":
    state = np.array([
        [-1, -1, -1],
        [0,  0,  0],
        [1,  1,  1]
    ])

    state = Node(state)

    hnet = HexapawnNet()
    for _ in range(5):
        mcts(state, player=1, hnet=hnet)
        print(state.q_value, state.num_visits, state.children)
        print("----------------------")
