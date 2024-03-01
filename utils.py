import numpy as np


"""
    +-----+-----+-----+
 0  |     |     |     |
    +-----+-----+-----+
 1  |     |     |     |
    +-----+-----+-----+
 2  |     |     |     |
    +-----+-----+-----+
       0     1     2
"""

MOVE_INDEX = {
    # white forward
    0: [[2, 0], [1, 0]], 1: [[2, 1], [1, 1]], 2: [[2, 2], [1, 2]],
    3: [[1, 0], [0, 0]], 4: [[1, 1], [0, 1]], 5: [[1, 2], [0, 2]],
    # black forward
    6: [[0, 0], [1, 0]], 7: [[0, 1], [1, 1]], 8: [[0, 2], [1, 2]],
    9: [[1, 0], [2, 0]], 10: [[1, 1], [2, 1]], 11: [[1, 2], [2, 2]],
    # white captures
    12: [[2, 0], [1, 1]], 13: [[2, 1], [1, 0]], 14: [[2, 1], [1, 2]], 15: [[2, 2], [1, 1]],
    16: [[1, 0], [0, 1]], 17: [[1, 1], [0, 0]], 18: [[1, 1], [0, 2]], 19: [[1, 2], [0, 1]],
    # black captures
    20: [[0, 0], [1, 1]], 21: [[0, 1], [1, 0]], 22: [[0, 1], [1, 2]], 23: [[0, 2], [1, 1]],
    24: [[1, 0], [2, 1]], 25: [[1, 1], [2, 0]], 26: [[1, 1], [2, 2]], 27: [[1, 2], [2, 1]],
}


def get_input_from_state(state: np.ndarray, player: int):
    white = (state == 1).flatten().astype(int)
    black = (state == - 1).flatten().astype(int)
    player = np.array([player == 1] * 3).astype(int)
    return np.append([white, black], player)


def get_move_index(move):
    for i in MOVE_INDEX:
        if move == MOVE_INDEX[i]:
            return i


if __name__ == "__main__":
    state = np.array([
        [-1, -1, -1],
        [0, 1, 0],
        [1, 0, 1]
    ])
    inp = get_input_from_state(state, 1)
    print(inp)
