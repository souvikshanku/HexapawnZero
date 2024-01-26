from game import get_valid_moves, is_game_over, make_move


class Node:
    def __init__(self, state) -> None:
        self.state = state
        self.value = None
        self.children = []

    def __repr__(self):
        return f"Node({self.state}), value = {self.value}"

    def selection(self):
        optimal_value = - float("inf")

        if len(self.children) > 0:
            for child in self.children:
                if child.value is None:
                    return child
                if child.value > optimal_value:
                    optimal_child, optimal_value = child, child.value

            return optimal_child.selection()

        return self

    def expansion(self, node, player):
        moves = get_valid_moves(node.state)
        for m in moves:
            child = make_move(node.state, m[0], m[1], player)
        self.children.append(child)

    def simulation(self):
        pass

    def backpropagation(self):
        pass


if __name__ == "__main__":
    root = Node([0] * 9)
    child1 = Node([1] + [0] * 8)
    child1.value = 1
    child2 = Node([0] * 8 + [1])
    child2.value = 2
    root.children = [child1, child2]
    print(root.selection())
