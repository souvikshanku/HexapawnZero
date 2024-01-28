import numpy as np

from game import is_game_over, get_valid_moves, make_move, draw_board


class Node:
    def __init__(self, state) -> None:
        self.state = np.array(state).reshape(3, 3)
        self.num_games = 0
        self.num_wins = 0
        self.value = 0
        self.parent = None
        self.children = []

    def __repr__(self):
        return f"Node({self.state.flatten()}, value = {self.value})"

    def selection(self, path):
        path.append(self)
        optimal_value = - float("inf")

        if len(self.children) > 0:
            for child in self.children:
                if child.value is None:
                    return child, path
                if child.value > optimal_value:
                    optimal_child, optimal_value = child, child.value

            return optimal_child.selection(path)

        return self, path

    def expansion(self, node, player):
        moves = get_valid_moves(node.state)
        for m in moves:
            child = Node(make_move(node.state, m[0], m[1], player))
            child.parent = self
        self.children.append(child)

    def simulation(self, player):
        num_wins = 0
        num_games = 0
        current_player = player

        while num_games < 5:
            state = self.state.copy()

            while not is_game_over(state, player)[0]:
                moves = get_valid_moves(state, player)
                random_move = moves[np.random.choice(len(moves))]
                state = make_move(
                    state=state,
                    _from=random_move[0],
                    _to=random_move[1],
                    player=player
                )
                draw_board(state)
                player = - player

            winning_player = is_game_over(state, player)[1]
            print(f"------------Game Over [{winning_player}]------------")
            if current_player == winning_player:
                num_wins += 1
            num_games += 1

        print(num_wins, "/", num_games)
        return num_wins, num_games

    def backpropagation(self, path, num_wins, num_games):
        for node in path[::-1]:
            node.num_games += num_games
            node.num_wins += num_wins
            node.value = node.num_wins / node.num_games

        return


if __name__ == "__main__":
    state = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    root = Node(state.flatten())
    # root.simulation(1)
    child1 = Node([-1] + [0] * 8)
    child1.value = 1
    child2 = Node([0] * 8 + [-1])
    child2.value = 2
    root.children = [child1, child2]
    child21 = Node([-1] * 9)
    child2.children = [child21]
    path = []
    path = root.selection(path)[1]
    # print(path)
    root.backpropagation(path, 1, 2)

    def dfs(node):
        for c in node.children:
            print(c)
            dfs(c)

    dfs(root)
