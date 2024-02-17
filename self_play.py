import copy

import numpy as np

from game import get_valid_moves, is_game_over, make_move
from mcts import mcts, Node
from model import HexapawnNet
from utils import MOVE_INDEX, get_input_from_state


def generate_examples(hnet: HexapawnNet):
    training_examples = []
    num_sims = 10
    num_episodes = 50

    for _ in range(num_episodes):
        state = Node(np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
        player = -1  # when multipllied by -1, it becomes 1 for first move
        examples = []

        while True:
            player = player * -1
            for _ in range(num_sims):
                mcts(state, player=player, hnet=hnet)

            action, target_policy = _get_action_from_policy(state)
            examples.append([
                state,
                target_policy,
                None
            ])

            next_state = make_move(state.state, _from=action[0], _to=action[1], player=player)
            state = Node(next_state)

            if is_game_over(state.state, player * -1)[0]:
                examples = _assign_rewards(examples, winner=player)
                training_examples += examples
                break

    return training_examples


def _get_action_from_policy(state: Node):
    policy = state.policy.detach().numpy().copy()
    valid_moves = get_valid_moves(state.state, state.player)

    # Mask illegal move
    for idx in MOVE_INDEX:
        if MOVE_INDEX[idx] not in valid_moves:
            policy[idx] = 0

    action_idx = np.random.choice(range(len(state.policy)), p=policy/sum(policy))

    return MOVE_INDEX[action_idx], policy / policy.sum()


def _assign_rewards(examples, winner):
    for i in range(len(examples)):
        if examples[i][0].player == winner:
            examples[i][2] = 1
        else:
            examples[i][2] = -1

        board_state = get_input_from_state(examples[i][0].state, examples[i][0].player)
        examples[i][0] = board_state

    return examples


def pit_nns(hnet1: HexapawnNet, hnet2: HexapawnNet):
    num_sims = 5
    num_episodes = 20
    hnet2_wins = 0

    def player_hnet(player, episode_count):
        # hnet2 plays as black for first 5 episodes
        if episode_count < num_episodes / 2:
            if player == 1:
                return hnet1
            else:
                return hnet2
        # hnet2 plays as white for last 5 episodes
        elif player == 1:
            return hnet2
        else:
            return hnet1

    for i in range(num_episodes):
        state = Node(np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
        player = -1  # when multipllied by -1, it becomes 1 for first move

        while True:
            player = player * -1
            for _ in range(num_sims):
                mcts(state, player=player, hnet=player_hnet(player, i))

            action, _ = _get_action_from_policy(state)
            next_state = make_move(state.state, _from=action[0], _to=action[1], player=player)
            state = Node(next_state)

            if is_game_over(state.state, player * -1)[0]:
                winner = is_game_over(state.state, player * -1)[1]

                if i < num_episodes / 2 and winner == -1:
                    hnet2_wins += 1
                elif i >= num_episodes / 2 and winner == 1:
                    hnet2_wins += 1

                break

    return hnet2_wins / num_episodes


def self_play(num_iters):
    hnet = HexapawnNet()

    for _ in range(num_iters):
        examples = generate_examples(hnet)
        # print(examples[-1])
        new_hnet = copy.deepcopy(hnet)
        new_hnet.train(examples)

        frac_win = pit_nns(hnet, new_hnet)
        print("frac_win: ", frac_win, "\n------------------")

        if frac_win > 0.5:
            hnet = new_hnet

    return hnet


if __name__ == "__main__":
    # hnet = HexapawnNet()
    # examples = generate_examples(hnet)

    # print(examples[0])
    # print(len(examples))

    # hnet1 = HexapawnNet()
    # hnet2 = HexapawnNet()

    # for i in range(10):
    #     print(pit_nns(hnet1, hnet2))

    trained_hnet = self_play(50)
    print("Final Score:", pit_nns(HexapawnNet(), trained_hnet))
