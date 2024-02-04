from game import is_game_over
from model import HexapawnNet
from node import Node
from utils import get_input_from_state


class MCTS:
    def __init__(self, state: Node, hnet: HexapawnNet) -> None:
        self.state = state
        self.hnet = hnet
        self.examples = {}  # (state, player): [policy, value]

    def search(self, player):
        if is_game_over(self.state)[0]:
            return - is_game_over(self.state)[1]

        self.state, path = self.state.selection()
        self.state.expansion(path[-1], player)

        inp = get_input_from_state(path[-1].state)
        pi, v = self.hnet.forward(inp)
        self.state.backpropagation(path, v)

        self.examples[inp] = [pi, None]
        return


"""
state = Start_Position
hnet = init_nn()
training_examples = []

mcts = MCTS(state, hnet)

for _ in range(num_sims):
    mcts.search()

    for example in mcts.examples:
        training_examples.append(example)

"""
