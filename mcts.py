import numpy as np

from game import is_game_over, get_move_idx
from utils import get_input_from_state


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
