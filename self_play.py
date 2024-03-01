import copy

import numpy as np

from game import is_game_over, make_move
from mcts import MCTS
from model import HexapawnNet
from utils import MOVE_INDEX, get_input_from_state


def generate_examples(hnet: HexapawnNet, num_episodes=1):
    training_examples = []
    num_sims = 10

    for _ in range(num_episodes):
        state = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        player = -1  # when multipllied by -1, it becomes 1 for first move
        examples = []

        while True:
            player = player * -1
            mcts = MCTS(hnet)

            for _ in range(num_sims):
                mcts.search(state, player=player)

            s = get_input_from_state(state, player)
            improved_policy = _get_mcts_policy(s, mcts.Nsa)
            examples.append([
                s,
                improved_policy,
                None,
            ])

            if not is_game_over(state, player * -1)[0]:
                action = _get_action_from_policy(improved_policy)
                state = make_move(state, _from=action[0], _to=action[1], player=player)

            else:
                examples = _assign_rewards(examples[:-1], winner=player)
                training_examples += examples
                break

    return training_examples


def _get_mcts_policy(state, Nsa):
    policy = np.zeros(28)
    for a in Nsa:
        s = eval(eval(a)[0].replace(" ", ","))

        if (state == s).all():
            policy[eval(a)[1]] = Nsa[a]

    if sum(policy) != 0:
        return policy / sum(policy)


def _get_action_from_policy(policy):
    action_idx = np.random.choice(range(len(policy)), p=policy)
    return MOVE_INDEX[action_idx]


def _assign_rewards(examples: list, winner: int):
    indc = 0 if winner == -1 else 1

    for ex in examples:
        if ex[0][-1] == indc:
            ex[2] = 1
        else:
            ex[2] = - 1

    return examples


def pit_nns(hnet1: HexapawnNet, hnet2: HexapawnNet, num_episodes=20):
    num_sims = 10
    hnet2_wins = 0
    win_as_black = 0
    win_as_white = 0

    def player_hnet(player, episode_count):
        # hnet2 plays as black for first 10 episodes
        if episode_count < num_episodes / 2:
            if player == 1:
                return hnet1
            else:
                return hnet2
        # hnet2 plays as white for last 10 episodes
        elif player == 1:
            return hnet2
        else:
            return hnet1

    for i in range(num_episodes):
        state = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        player = -1  # when multipllied by -1, it becomes 1 for first move

        while True:
            player = player * -1
            mcts = MCTS(player_hnet(player, i))

            for _ in range(num_sims):
                mcts.search(state, player=player)

            s = str(get_input_from_state(state, player))
            action = _get_action_from_policy(mcts.Ps[s])
            state = make_move(state, _from=action[0], _to=action[1], player=player)

            if is_game_over(state, player * -1)[0]:
                if i < num_episodes / 2 and player == -1:
                    win_as_black += 1
                    hnet2_wins += 1

                elif i >= num_episodes / 2 and player == 1:
                    win_as_white += 1
                    hnet2_wins += 1

                break

    print(f"As black: {win_as_black} out of {num_episodes // 2}")
    print(f"As white: {win_as_white} out of {num_episodes // 2}")
    print("Total Wins: ", hnet2_wins)
    return hnet2_wins / num_episodes


def self_play(num_iters: int):
    hnet = HexapawnNet()
    examples = []

    for _ in range(num_iters):
        examples += generate_examples(hnet, 10)

        new_hnet = copy.deepcopy(hnet)
        new_hnet.train(examples)

        frac_win = pit_nns(hnet, new_hnet)
        print("frac_win: ", frac_win,)

        if frac_win >= 0.5:
            hnet = new_hnet
            examples = []

    return hnet


if __name__ == "__main__":
    # import torch

    trained_hnet = self_play(20)
    # torch.save(trained_hnet, "./model.bin")
    hnet = HexapawnNet()
    print("--------------------------\n""Final Score with random:", pit_nns(hnet, trained_hnet, 50))
