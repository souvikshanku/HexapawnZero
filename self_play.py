from mcts import mcts

import numpy as np

from game import get_valid_moves, is_game_over, make_move
from utils import MOVE_INDEX
from mcts import Node
from model import HexapawnNet


def generate_examples(hnet: HexapawnNet):
    training_examples = []
    num_sims = 5
    num_episodes = 15

    for _ in range(num_episodes):
        state = Node(np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
        player = -1  # when multipllied by -1, it becomes 1 for first move
        examples = []

        while True:
            player = player * -1
            for _ in range(num_sims):
                mcts(state, player=player, hnet=hnet)

            examples.append([state, state.policy, None])
            action = _get_action_from_policy(state)
            state = Node(make_move(state, _from=action[0], _to=action[1], player=player * -1))

            if is_game_over(state, player * -1)[0]:
                examples = _assign_rewards(examples, player)
                training_examples.append(examples)
                break


def _get_action_from_policy(state: Node):
    policy = state.policy.detach().numpy()
    valid_moves = get_valid_moves(state.state, state.player)

    for idx in MOVE_INDEX:
        if MOVE_INDEX[idx] not in valid_moves:
            policy[idx] = 0

    action_idx = np.random.choice(range(len(state.policy)), p=policy/sum(policy))

    return MOVE_INDEX[action_idx]


def _assign_rewards(examples, player):
    pass
