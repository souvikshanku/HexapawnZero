import numpy as np
import torch

from game import is_game_over, get_move_idx, get_valid_moves, make_move
from utils import get_input_from_state  # , MOVE_INDEX
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

    def __str__(self) -> str:
        return f"{str(self.state.flatten())} {str(self.player)}"

    def __repr__(self):
        return f"Node({self.state.flatten()}, player = {float(self.player)})"

    def expand(self, player):
        moves = get_valid_moves(self.state, player)
        for m in moves:
            child_state = make_move(self.state.copy(), m[0], m[1], player)
            child = Node(child_state)
            child.parent = self
            self.children.append(child)

        return self.children

    def get_mcts_policy(self, player):
        policy = torch.zeros(28)
        total_visits = sum([child.num_visits for child in self.children])
        for child in self.children:
            idx = get_move_idx(self, child, player)
            policy[idx] = child.num_visits / total_visits

        return policy


class MCTS:
    def __init__(self) -> None:
        self.tree = {
            # str(state): {num_visits: 0, q_value: 0, policy: []]
        }

    def _add_to_tree(self, state):
        self.tree[str(state)] = {
            "num_visits": 0, "q_value": 0, "policy": []
        }

    def search(self, state: Node, player: int, hnet: HexapawnNet):
        state.player = player

        game_over, reward = is_game_over(state.state, player)
        if game_over:
            return - reward

        if str(state) not in self.tree:
            p, v = hnet.predict(get_input_from_state(state.state, player))
            self._add_to_tree(state)
            self.tree[str(state)]["num_visits"] = 1
            self.tree[str(state)]["policy"] = torch.exp(p)
            return - v

        max_u, best_child = -float("inf"), -1

        for c in state.expand(player):
            idx = get_move_idx(state, c, player)
            p_sa = self.tree[str(state)]["policy"][idx]
            n_s = self.tree[str(state)]["num_visits"]

            if str(c) in self.tree:
                n_sa = self.tree[str(c)]["num_visits"]
                q_sa = self.tree[str(c)]["q_value"]
            else:
                n_sa = 0
                q_sa = 0

            u = q_sa + 1 * p_sa * np.sqrt(n_s) / (1 + n_sa)

            if u > max_u:
                max_u = u
                best_child = c

        v = self.search(best_child, - player, hnet)

        if str(best_child) in self.tree:
            self.tree[str(best_child)]["q_value"] = (n_sa * q_sa + v) / (n_sa + 1)
            self.tree[str(best_child)]["num_visits"] += 1
        else:
            self.tree[str(best_child)] = {
                "num_visits": 1, "q_value": v, "policy": []
            }

        self.tree[str(state)]["num_visits"] += 1

        return - v


if __name__ == "__main__":
    hnet = HexapawnNet()
    mcts = MCTS()

    state = np.array([
        [-1, -1, -1],
        [0,  0,  0],
        [1,  1,  1]
    ])

    node = Node(state)

    for i in range(5):
        print(mcts.search(node, 1, hnet))
