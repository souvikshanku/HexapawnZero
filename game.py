import numpy as np


BOARD = list("""
    +-----+-----+-----+
 0  |     |     |     |
    +-----+-----+-----+
 1  |     |     |     |
    +-----+-----+-----+
 2  |     |     |     |
    +-----+-----+-----+
       0     1     2
""")


def draw_board(state: np.ndarray) -> None:
    board = BOARD.copy()

    pos = [32, 38, 44, 80, 86, 92, 128, 134, 140]
    state = state.flatten()
    for i, move in enumerate(state):
        if move == 1:
            board[pos[i]] = "W"
        elif move == -1:
            board[pos[i]] = "B"

    print("".join(board))


def get_valid_moves(state: np.ndarray, player: int) -> list:
    pieces_pos = []
    for row in range(len(state)):
        for col in range(len(state)):
            if state[row, col] == player:
                pieces_pos.append([row, col])

    valid_moves = []  # with element of this form: (from, to)

    for pos in pieces_pos:
        x, y = pos[0], pos[1]

        if player == 1:
            if x == 0:
                continue
            else:
                if state[x - 1, y] == 0:  # If no black piece up
                    valid_moves.append([pos, [x - 1, y]])
                # Capture
                if y - 1 >= 0 and state[x - 1, y - 1] == -1:
                    valid_moves.append([pos, [x - 1, y - 1]])
                if y + 1 < 3 and state[x - 1, y + 1] == -1:
                    valid_moves.append([pos, [x - 1, y + 1]])
        else:
            if x == -1:
                continue
            else:
                if state[x + 1, y] == 0:  # If no white piece down
                    valid_moves.append([pos, [x + 1, y]])
                # Capture
                if y - 1 >= 0 and state[x + 1, y - 1] == 1:
                    valid_moves.append([pos, [x + 1, y - 1]])
                if y + 1 < 3 and state[x + 1, y + 1] == 1:
                    valid_moves.append([pos, [x + 1, y + 1]])

    return valid_moves


def is_game_over(state: np.ndarray, player: int) -> tuple[bool, int]:
    if 1 in state[0]:
        return True, 1
    elif -1 in state[2]:
        return True, -1

    valid_moves = get_valid_moves(state, player)
    if not valid_moves:
        if player == 1:
            return True, -1
        else:
            return True, 1

    return False, None


def make_move(
        state: np.ndarray, _from: list, _to: list, player: int
) -> np.ndarray:
    state[_from[0], _from[1]] = 0
    state[_to[0], _to[1]] = player
    return state


if __name__ == "__main__":
    state = np.array([
        [-1, -1, -1],
        [0, 1, 0],
        [1, 0, 1]
    ])
    draw_board(state)
    print(get_valid_moves(state, 1))
    print(is_game_over(state, 1))
    state_after = make_move(state, [1, 1], [0, 0], 1)
    draw_board(state_after)
    print(is_game_over(state_after, 1))
