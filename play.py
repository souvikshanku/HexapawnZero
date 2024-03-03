import numpy as np
import torch

import os
import time

from game import draw_board, get_valid_moves, is_game_over, make_move
from mcts import MCTS
from model import HexapawnNet
from self_play import get_action_from_policy, learn_by_self_play


def ask_for_valid_move(state: np.ndarray, player: int) -> list:
    while True:
        moves = get_valid_moves(state, player)
        move_d = {i: move for i, move in enumerate(moves)}
        print("Valid moves:")
        for k in move_d:
            print(f"{move_d[k][0]} --> {move_d[k][1]}    {k}")

        idx = int(input("Your move? "))

        if idx in move_d.keys():
            return move_d[idx]
        else:
            print("Please provide a valid move.")


def choose_move(state: np.ndarray, player: int, hnet: HexapawnNet) -> np.ndarray:
    mcts = MCTS(hnet)
    num_sims = 5

    for _ in range(num_sims):
        mcts.search(state, player=player)

    action = get_action_from_policy(state, player, mcts.Nsa)
    state = make_move(state, _from=action[0], _to=action[1], player=player)

    return state


def render(state: np.ndarray) -> None:
    os.system("clear")
    print(draw_board(state))


if __name__ == "__main__":
    try:
        hnet = torch.load("model.bin")
    except Exception:
        print("Trained model could not be found. Training HexapawnZero now...\n")
        hnet = learn_by_self_play(10)
        torch.save(hnet, "model.bin")
        print("Training completed!\n")

    player = int(input("Do you wanna go first? - 1 (yes), 2 (no)\n"))
    os.system("clear")

    if player == 1:
        state = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        render(state)

    else:
        player = - 1
        state = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        render(state)
        time.sleep(2)
        state = choose_move(state, player * - 1, hnet)
        render(state)

    while True:
        if not is_game_over(state, player)[0]:
            move = ask_for_valid_move(state, player)
            state = make_move(state, move[0], move[1], player)
            render(state)
        else:
            break
        if not is_game_over(state, player)[0]:
            time.sleep(3)
            state = choose_move(state, player * -1, hnet)
            render(state)
        else:
            break

    reward = is_game_over(state, player)[1]

    if reward == 1:
        print("You win!")
    else:
        print("HexapawnZero wins!")
